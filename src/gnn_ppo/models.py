from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    def __init__(self, shape=()) -> None:
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = x.reshape(-1, *self.mean.shape)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype="float64")
        self.var = np.asarray(state["var"], dtype="float64")
        self.count = float(state["count"])


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    hidden_layers: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class PaperGraphGCN(nn.Module):
    """
    Paper Section 5.1 graph encoder:
    H^(l+1)=sigma(D_hat^-1/2 A_hat D_hat^-1/2 H^(l) W^(l)), then mean pooling.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        use_layer_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        input_dim = node_feature_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.norms.append(nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity())
            input_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, node_features: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        h = node_features
        adjacency = normalized_adjacency.to(h.device)
        for layer, norm in zip(self.layers, self.norms):
            h = torch.matmul(adjacency, h)
            h = layer(h)
            h = self.dropout(torch.relu(norm(h)))
        return h.mean(dim=1)


class EdgeWeightedBiGNN(nn.Module):
    """Legacy fallback encoder; not used for paper-aligned experiments."""

    def __init__(
        self,
        num_start: int,
        num_end: int,
        hidden_dim: int = 128,
        *,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_start = num_start
        self.num_end = num_end
        self.hidden_dim = hidden_dim

        self.emb_start = nn.Embedding(num_start, hidden_dim)
        self.emb_end = nn.Embedding(num_end, hidden_dim)

        self.w_s1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias_end1 = nn.Parameter(torch.zeros(hidden_dim))
        self.ln_end1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        self.w_e1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias_start1 = nn.Parameter(torch.zeros(hidden_dim))
        self.ln_start1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        self.w_s2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias_end2 = nn.Parameter(torch.zeros(hidden_dim))
        self.ln_end2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        self.w_e2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias_start2 = nn.Parameter(torch.zeros(hidden_dim))
        self.ln_start2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.bias_end1)
        nn.init.zeros_(self.bias_start1)
        nn.init.zeros_(self.bias_end2)
        nn.init.zeros_(self.bias_start2)

    def forward(self, edge_weights: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        device = self.emb_start.weight.device
        batch_size, _num_edges = edge_weights.shape
        src_idx = src_idx.to(device)
        dst_idx = dst_idx.to(device)
        edge_weights = edge_weights.to(device)

        h_start = self.emb_start.weight.unsqueeze(0).expand(batch_size, -1, -1)
        h_end = self.emb_end.weight.unsqueeze(0).expand(batch_size, -1, -1)
        edge_scale = edge_weights.unsqueeze(-1)

        src_h = h_start[:, src_idx, :]
        msg_s = self.w_s1(src_h) * edge_scale
        new_h_end = torch.zeros(batch_size, self.num_end, self.hidden_dim, device=device)
        new_h_end.scatter_add_(1, dst_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.hidden_dim), msg_s)
        new_h_end = self.dropout(self.act(self.ln_end1(new_h_end + self.bias_end1)))
        h_end = h_end + new_h_end

        dst_h = h_end[:, dst_idx, :]
        msg_e = self.w_e1(dst_h) * edge_scale
        new_h_start = torch.zeros(batch_size, self.num_start, self.hidden_dim, device=device)
        new_h_start.scatter_add_(
            1, src_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.hidden_dim), msg_e
        )
        new_h_start = self.dropout(self.act(self.ln_start1(new_h_start + self.bias_start1)))
        h_start = h_start + new_h_start

        src_h2 = h_start[:, src_idx, :]
        msg_s2 = self.w_s2(src_h2) * edge_scale
        new_h_end2 = torch.zeros(batch_size, self.num_end, self.hidden_dim, device=device)
        new_h_end2.scatter_add_(
            1, dst_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.hidden_dim), msg_s2
        )
        new_h_end2 = self.dropout(self.act(self.ln_end2(new_h_end2 + self.bias_end2)))
        h_end = h_end + new_h_end2

        dst_h2 = h_end[:, dst_idx, :]
        msg_e2 = self.w_e2(dst_h2) * edge_scale
        new_h_start2 = torch.zeros(batch_size, self.num_start, self.hidden_dim, device=device)
        new_h_start2.scatter_add_(
            1, src_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.hidden_dim), msg_e2
        )
        new_h_start2 = self.dropout(self.act(self.ln_start2(new_h_start2 + self.bias_start2)))
        h_start = h_start + new_h_start2

        return torch.cat([h_start.mean(dim=1), h_end.mean(dim=1)], dim=-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        graph: dict[str, Any],
        *,
        action_dim: int,
        context_dim: int,
        gnn_hidden: int,
        gnn_layers: int,
        mlp_hidden: int,
        mlp_layers: int,
        use_layer_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.graph_mode = str(graph.get("mode", "legacy_bipartite"))
        if self.graph_mode == "paper_multimodal":
            self.register_buffer("normalized_adjacency", graph["normalized_adjacency"].clone().detach())
            self.gnn = PaperGraphGCN(
                int(graph["node_feature_dim"]),
                gnn_hidden,
                gnn_layers,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            graph_embedding_dim = gnn_hidden
        else:
            self.register_buffer("src_idx", graph["src_idx"].clone().detach())
            self.register_buffer("dst_idx", graph["dst_idx"].clone().detach())
            self.gnn = EdgeWeightedBiGNN(
                int(graph["num_start"]),
                int(graph["num_end"]),
                hidden_dim=gnn_hidden,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            graph_embedding_dim = 2 * gnn_hidden

        actor_input_dim = graph_embedding_dim + context_dim
        self.actor = make_mlp(
            actor_input_dim,
            mlp_hidden,
            action_dim,
            hidden_layers=mlp_layers,
            dropout=dropout,
        )
        self.critic = make_mlp(
            actor_input_dim,
            mlp_hidden,
            1,
            hidden_layers=mlp_layers,
            dropout=dropout,
        )

    def forward(self, graph_state_batch: torch.Tensor, context_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if graph_state_batch.dim() == 1:
            graph_state_batch = graph_state_batch.unsqueeze(0)
        if context_batch.dim() == 1:
            context_batch = context_batch.unsqueeze(0)
        if self.graph_mode == "paper_multimodal":
            graph_embedding = self.gnn(graph_state_batch, self.normalized_adjacency)
        else:
            graph_embedding = self.gnn(graph_state_batch, self.src_idx, self.dst_idx)
        encoded = torch.cat([graph_embedding, context_batch.to(graph_embedding.device)], dim=-1)
        return self.actor(encoded), self.critic(encoded)

    def get_action_and_value(
        self,
        graph_state: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(graph_state, context)
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask.to(dtype=torch.bool, device=logits.device), -1e9)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, log_prob, entropy, value


class PPOAgent:
    def __init__(
        self,
        graph: dict[str, Any],
        model_config: dict[str, Any],
        device: torch.device,
        *,
        action_dim: int,
        context_dim: int,
    ) -> None:
        self.graph = graph
        self.config = model_config
        self.device = device
        self.ac = ActorCritic(
            graph,
            action_dim=action_dim,
            context_dim=context_dim,
            gnn_hidden=int(model_config["gnn_hidden"]),
            gnn_layers=int(model_config.get("gnn_layers", 2)),
            mlp_hidden=int(model_config["mlp_hidden"]),
            mlp_layers=int(model_config.get("mlp_layers", 2)),
            use_layer_norm=bool(model_config["use_layer_norm"]),
            dropout=float(model_config["dropout"]),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=float(model_config["learning_rate"]))
        if str(graph.get("mode")) == "paper_multimodal":
            rms_shape = (int(graph["num_nodes"]), int(graph["node_feature_dim"]))
        else:
            rms_shape = (int(graph["src_idx"].shape[0]),)
        self.state_rms = RunningMeanStd(shape=rms_shape)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return self.state_rms.normalize(state).astype(np.float32)

    def update_state_stats(self, state: np.ndarray) -> None:
        self.state_rms.update(state)

    def choose_action(
        self,
        graph_state: np.ndarray,
        context: np.ndarray,
        action_mask: np.ndarray,
    ) -> tuple[int, float, float]:
        norm_state = self.normalize_state(graph_state)
        with torch.no_grad():
            graph_state_t = torch.FloatTensor(norm_state).to(self.device)
            context_t = torch.FloatTensor(context).to(self.device)
            mask_t = torch.BoolTensor(action_mask).to(self.device)
            action, log_prob, _entropy, value = self.ac.get_action_and_value(
                graph_state_t, context_t, action_mask=mask_t
            )
        return action.cpu().item(), log_prob.cpu().item(), value.cpu().item()

    def greedy_action(self, graph_state: np.ndarray, context: np.ndarray, action_mask: np.ndarray) -> int:
        norm_state = self.normalize_state(graph_state)
        with torch.no_grad():
            graph_state_t = torch.FloatTensor(norm_state).to(self.device)
            context_t = torch.FloatTensor(context).to(self.device)
            mask_t = torch.BoolTensor(action_mask).to(self.device)
            logits, _ = self.ac.forward(graph_state_t, context_t)
            if mask_t.dim() == 1:
                mask_t = mask_t.unsqueeze(0)
            logits = logits.masked_fill(~mask_t.to(dtype=torch.bool, device=logits.device), -1e9)
        return torch.argmax(logits, dim=-1).cpu().item()

    def learn(self, trajectories: list[dict[str, Any]]) -> dict[str, float]:
        states = np.array([t["state"] for t in trajectories])
        contexts = np.array([t["context"] for t in trajectories])
        action_masks = np.array([t["action_mask"] for t in trajectories])
        actions = np.array([t["action"] for t in trajectories])
        old_log_probs = np.array([t["log_prob"] for t in trajectories])
        rewards = np.array([t["reward"] for t in trajectories])
        dones = np.array([t["done"] for t in trajectories])
        next_states = np.array([t["next_state"] for t in trajectories])
        next_contexts = np.array([t["next_context"] for t in trajectories])

        for state in states:
            self.update_state_stats(state)
        for next_state in next_states:
            self.update_state_stats(next_state)

        states_t = torch.FloatTensor(self.normalize_state(states)).to(self.device)
        contexts_t = torch.FloatTensor(contexts).to(self.device)
        action_masks_t = torch.BoolTensor(action_masks).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        next_states_t = torch.FloatTensor(self.normalize_state(next_states)).to(self.device)
        next_contexts_t = torch.FloatTensor(next_contexts).to(self.device)

        gamma = float(self.config["gamma"])
        gae_lambda = float(self.config["gae_lambda"])
        with torch.no_grad():
            _, values_t = self.ac.forward(states_t, contexts_t)
            values_np = values_t.squeeze(-1).cpu().numpy()
            _, next_values_t = self.ac.forward(next_states_t, next_contexts_t)
            next_values_np = next_values_t.squeeze(-1).cpu().numpy()

        advantages = np.zeros_like(rewards, dtype=float)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0 if dones[t] else next_values_np[t]
            else:
                next_value = 0.0 if dones[t] else values_np[t + 1]
            delta = rewards[t] + gamma * next_value - values_np[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae
        returns = advantages + values_np
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        dataset_size = len(states_t)
        indices = np.arange(dataset_size)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        n_batches = 0

        for _ in range(int(self.config["ppo_epochs"])):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, int(self.config["batch_size"])):
                batch_indices = indices[start : start + int(self.config["batch_size"])]
                batch_states = states_t[batch_indices]
                batch_contexts = contexts_t[batch_indices]
                batch_action_masks = action_masks_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]

                _, new_log_probs, entropy, new_values = self.ac.get_action_and_value(
                    batch_states,
                    batch_contexts,
                    action=batch_actions,
                    action_mask=batch_action_masks,
                )
                new_values = new_values.squeeze(-1)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - float(self.config["clip_eps"]),
                    1 + float(self.config["clip_eps"]),
                ) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                total_loss = (
                    actor_loss
                    + float(self.config["value_loss_coef"]) * critic_loss
                    + float(self.config["entropy_coef"]) * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), float(self.config["max_grad_norm"]))
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_frac += (torch.abs(ratio - 1) > float(self.config["clip_eps"])).float().mean().item()
                n_batches += 1

        return {
            "actor_loss": total_actor_loss / n_batches,
            "critic_loss": total_critic_loss / n_batches,
            "entropy": total_entropy / n_batches,
            "clip_frac": total_clip_frac / n_batches,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "model_state": self.ac.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "state_rms": self.state_rms.state_dict(),
            "model_config": self.config,
            "graph_mode": self.graph.get("mode"),
        }

    def load_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        self.ac.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.state_rms.load_state_dict(checkpoint["state_rms"])
