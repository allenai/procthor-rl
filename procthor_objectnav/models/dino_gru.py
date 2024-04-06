"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""

from typing import Optional, List, Dict, cast, Tuple

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Dict as SpaceDict

from allenact.embodiedai.models.visual_nav_models import (
    VisualNavActorCritic,
    FusionType,
)
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.algorithms.onpolicy_sync.policy import (
    ObservationType,
    DistributionType,
)
from ..utils.utils import log_ac_return

class DinoTensorNavActorCritic(VisualNavActorCritic):
    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        add_prev_actions=False,
        add_prev_action_null_token=False,
        action_embed_size=6,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # custom params
        rgb_dino_preprocessor_uuid: Optional[str] = None,
        goal_dims: int = 32,
        dino_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        visualize: bool = False,
        task_id_uuid: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
            **kwargs,
        )

        self.visualize = visualize
        self.task_id_uuid = task_id_uuid

        if rgb_dino_preprocessor_uuid is not None:
            dino_preprocessor_uuid = rgb_dino_preprocessor_uuid
            self.goal_visual_encoder = DinoTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                dino_preprocessor_uuid,
                goal_dims,
                dino_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.goal_visual_encoder(observations)

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        actor_critic_output, memory = super().forward(observations, memory, prev_actions, masks)
        if self.visualize:
            log_ac_return(actor_critic_output, observations[self.task_id_uuid])
        return actor_critic_output, memory


class DinoTensorNavActorCriticCodebook(DinoTensorNavActorCritic):
    def __init__(
            # base params
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            hidden_size=512,
            multiple_beliefs=False,
            beliefs_fusion: Optional[FusionType] = None,
            auxiliary_uuids: Optional[List[str]] = None,
            add_prev_actions=False,
            # codebook params
            codebook_type: str = "learned",
            codebook_indexing: str = "softmax",
            codebook_size: int = 256,
            codebook_dim: int = 10,
            codebook_temperature: float = 1.0,
            codebook_dropout_prob: float = 0.1,
            codebook_embeds_type: str = "joint_embeds",
            codebook_topk: int = 16,
            **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
            add_prev_actions=add_prev_actions,
            **kwargs,
        )

        self.codebook_type = codebook_type
        self.codebook_indexing = codebook_indexing
        self.codebook_temperature = codebook_temperature
        self.dropout_prob = codebook_dropout_prob
        self.codebook_embeds_type = codebook_embeds_type
        self.codebook_topk = codebook_topk

        self.create_codebook(
            codebook_size,
            codebook_dim,
            codebook_embeds_type,
            add_prev_actions,
        )
        self.train()

    def create_codebook(
        self,
        codebook_size,
        codebook_dim,
        codebook_embeds_type,
        add_prev_actions,
    ):
        self.codebook = torch.nn.Parameter(torch.randn(codebook_size, codebook_dim))
        self.codebook.requires_grad = True

        # dropout to prevent codebook collapse
        self.dropout = nn.Dropout(self.dropout_prob)

        # codebook indexing
        if codebook_embeds_type == "joint_embeds":
            embed_size = self.goal_visual_encoder.output_dims
            if add_prev_actions:
                embed_size += self.prev_action_embedder.output_size

            self.linear_codebook_indexer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(embed_size, codebook_size),
            )
            self.linear_upsample = nn.Sequential(
                nn.Linear(codebook_dim, embed_size),
            )
        elif codebook_embeds_type == "beliefs":
            self.linear_codebook_indexer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self._hidden_size, codebook_size),
            )
            self.linear_upsample = nn.Sequential(
                nn.Linear(codebook_dim, self._hidden_size),
            )
        else:
            raise NotImplementedError

    def forward_codebook(self, embeds: torch.Tensor) -> torch.Tensor:
        if self.codebook_indexing == "gumbel_softmax":
            codebook_probs = F.gumbel_softmax(self.linear_codebook_indexer(embeds),
                                              tau=self.codebook_temperature, hard=True, dim=-1)
        elif self.codebook_indexing == "softmax":
            codebook_probs = F.softmax(self.linear_codebook_indexer(embeds), dim=-1)
            if self.dropout_prob > 0:
                codebook_probs = self.dropout(codebook_probs)
        elif self.codebook_indexing == "topk_softmax":
            softmax_output = F.softmax(self.linear_codebook_indexer(embeds), dim=-1)
            if self.dropout_prob > 0:
                softmax_output = self.dropout(softmax_output)
            topk_values, topk_indices = torch.topk(softmax_output, self.codebook_topk, dim=-1)
            codebook_probs = torch.zeros_like(softmax_output)
            codebook_probs.scatter_(-1, topk_indices, topk_values)
        else:
            raise NotImplementedError("Codebook indexing method should be one of 'gumbel_softmax', 'softmax', 'topk_softmax'")

        if self.codebook_type == "learned":
            code_output = torch.einsum('nbm,md->nbd', codebook_probs, self.codebook)
        elif self.codebook_type == "random":
            code_output = torch.einsum('nbm,md->nbd', codebook_probs, self.codebook.detach())
        elif self.codebook_type == "binary":
            code_output = torch.einsum('nbm,md->nbd', codebook_probs, 2.0 * (self.codebook > 0.) - 1)
        else:
            raise NotImplementedError("Codebook type should be one of 'learned', 'random', 'binary'")

        return self.linear_upsample(code_output)

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        obs_embeds = self.forward_encoder(observations)

        # 1.2 use embedding model to get prev_action embeddings
        if self.prev_action_embedder.input_size == self.action_space.n + 1:
            # In this case we have a unique embedding for the start of an episode
            prev_actions_embeds = self.prev_action_embedder(
                torch.where(
                    condition=0 != masks.view(*prev_actions.shape),
                    input=prev_actions + 1,
                    other=torch.zeros_like(prev_actions),
                )
            )
        else:
            prev_actions_embeds = self.prev_action_embedder(prev_actions)
        joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)

        if self.codebook_embeds_type == "joint_embeds":
            ###########################
            # Codebook before RNN
            ###########################
            joint_embeds = self.forward_codebook(joint_embeds)

        # 2. use RNNs to get single/multiple beliefs
        beliefs_dict = {}
        for key, model in self.state_encoders.items():
            beliefs_dict[key], rnn_hidden_states = model(
                joint_embeds, memory.tensor(key), masks
            )
            memory.set_tensor(key, rnn_hidden_states)  # update memory here

        # 3. fuse beliefs for multiple belief models
        beliefs, task_weights = self.fuse_beliefs(
            beliefs_dict, obs_embeds
        )  # fused beliefs

        if self.codebook_embeds_type == "beliefs":
            ###########################
            # Codebook after RNN
            ###########################
            beliefs = self.forward_codebook(beliefs)

        # 4. prepare output
        extras = (
            {
                aux_uuid: {
                    "beliefs": (
                        beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs
                    ),
                    "obs_embeds": obs_embeds,
                    "aux_model": (
                        self.aux_models[aux_uuid]
                        if aux_uuid in self.aux_models
                        else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {}
        )

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        distributions = self.actor(beliefs)
        values = self.critic(beliefs)
        actor_critic_output = ActorCriticOutput(
            distributions=distributions,
            values=values,
            extras=extras,
        )

        if self.visualize:
            log_ac_return(actor_critic_output, observations[self.task_id_uuid])

        return actor_critic_output, memory



class DinoTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        dino_preprocessor_uuid: str,
        goal_embed_dims: int = 32,
        dino_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.dino_uuid = dino_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.dino_hid_out_dims = dino_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]
        if isinstance(self.goal_space, gym.spaces.Discrete):
            self.embed_goal = nn.Embedding(
                num_embeddings=self.goal_space.n,
                embedding_dim=self.goal_embed_dims,
            )
        elif isinstance(self.goal_space, gym.spaces.Box):
            self.embed_goal = nn.Linear(self.goal_space.shape[-1], self.goal_embed_dims)
        else:
            raise NotImplementedError

        self.blind = self.dino_uuid not in observation_spaces.spaces
        if not self.blind:
            self.dino_tensor_shape = observation_spaces.spaces[self.dino_uuid].shape
            self.dino_compressor = nn.Sequential(
                nn.Conv2d(self.dino_tensor_shape[-1], self.dino_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.dino_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.dino_hid_out_dims[1] + self.goal_embed_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return (
                self.combine_hid_out_dims[-1]
                * self.dino_tensor_shape[0]
                * self.dino_tensor_shape[1]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_dino(self, observations):
        return self.dino_compressor(observations[self.dino_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1, 1).expand(
            -1, -1, self.dino_tensor_shape[-3], self.dino_tensor_shape[-2]
        )

    def adapt_input(self, observations):
        observations = {**observations}
        dino = observations[self.dino_uuid]
        goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(dino.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = dino.shape[:3]
        else:
            nstep, nsampler = dino.shape[:2]

        observations[self.dino_uuid] = dino.view(-1, *dino.shape[-3:])
        observations[self.goal_uuid] = goal.view(-1, goal.shape[-1])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        embs = [
            self.compress_dino(observations),
            self.distribute_target(observations),
        ]
        x = self.target_obs_combiner(
            torch.cat(
                embs,
                dim=1,
            )
        )
        x = x.reshape(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)

