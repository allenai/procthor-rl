"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""

from collections import OrderedDict
from typing import Optional, List, Tuple, Dict, cast

import gym
import torch
import torch.nn as nn
from allenact.algorithms.onpolicy_sync.policy import ObservationType, DistributionType
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from allenact.embodiedai.models.visual_nav_models import (
    VisualNavActorCritic,
    FusionType,
)
from allenact.utils.model_utils import FeatureEmbedding
from allenact.utils.system import get_logger
from gym.spaces import Dict as SpaceDict

from procthor_objectnav.models.dino_gru import DinoTensorGoalEncoder
from procthor_objectnav.models.nn.transformer import (
    ModelArgs,
    PositionalEncoder,
    TransformerDecoder,
)


from ..utils.utils import log_ac_return


class DinoTransformerNavActorCritic(VisualNavActorCritic):
    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        num_tx_layers=3,
        num_tx_heads=8,
        add_prev_actions=True,
        add_prev_action_null_token=True,
        action_embed_size=7,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # custom params
        rgb_dino_preprocessor_uuid: Optional[str] = None,
        depth_preprocessor_uuid: Optional[str] = None,
        goal_dims: int = 32,
        dino_compressor_hidden_out_dims: Tuple[int, int] = (384, 512),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        max_steps: int = 1000,
        # max_steps_for_training: int = 128,
        time_step_uuid: Optional[str] = None,
        initial_tgt_cache_shape: Tuple[int, int] = (1024, 1024),  # seq, bs
        traj_idx_uuid: Optional[str] = None,
        use_transformer_encoder: bool = False,
        transformer_encoder_layers: int = 3,
        transformer_encoder_heads: int = 8,
        visualize: bool = False,
        task_id_uuid: Optional[str] = None,
        relevant_object_box_uuid: Optional[str] = None,
        concat_depth: bool = False,
        num_depth_channels: int = 1,
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

        self.time_step_counter = 0
        self.traj_idx_uuid = traj_idx_uuid
        self.visualize = visualize
        self.task_id_uuid = task_id_uuid
        self.relevant_object_box_uuid = relevant_object_box_uuid

        self.max_steps = max_steps
        # self.max_steps_for_training = max_steps_for_training
        self.time_step_uuid = time_step_uuid
        if rgb_dino_preprocessor_uuid is not None:
            dino_preprocessor_uuid = rgb_dino_preprocessor_uuid
            if use_transformer_encoder:
                self.goal_visual_encoder = DinoTxGoalEncoder(
                    self.observation_space,
                    goal_sensor_uuid,
                    dino_preprocessor_uuid,
                    depth_preprocessor_uuid,
                    relevant_object_box_uuid,
                    goal_embed_dims=hidden_size,
                    dino_compressor_hidden_out_dims=dino_compressor_hidden_out_dims,
                    combiner_hidden_out_dims=hidden_size,
                    combiner_layers=transformer_encoder_layers,
                    combiner_heads=transformer_encoder_heads,
                    concat_depth=concat_depth,
                    num_depth_channels=num_depth_channels,
                )
            else:
                self.goal_visual_encoder = DinoTensorGoalEncoder(
                    self.observation_space,
                    goal_sensor_uuid,
                    dino_preprocessor_uuid,
                    goal_dims,
                    dino_compressor_hidden_out_dims,
                    combiner_hidden_out_dims,
                )

        self.state_encoders_time: Optional[nn.ModuleDict] = None
        self.state_encoders_linear: Optional[nn.ModuleDict] = None
        self.state_encoders_text: Optional[nn.ModuleDict] = None
        self.create_tx_state_encoders(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            num_tx_layers=num_tx_layers,
            num_tx_heads=num_tx_heads,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
            initial_tgt_cache_shape=initial_tgt_cache_shape,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

    def create_tx_state_encoders(
        self,
        obs_embed_size: int,
        prev_action_embed_size: int,
        num_tx_layers: int,
        num_tx_heads: int,
        add_prev_actions: bool,
        add_prev_action_null_token: bool,
        initial_tgt_cache_shape: Tuple[int, int],
    ):
        tx_input_size = obs_embed_size
        self.prev_action_embedder = FeatureEmbedding(
            input_size=int(add_prev_action_null_token) + self.action_space.n,
            output_size=prev_action_embed_size if add_prev_actions else 0,
        )
        if add_prev_actions:  # concat add
            tx_input_size += prev_action_embed_size

        state_encoders_params = ModelArgs(
            dim=self._hidden_size,  # obs_embed_size,
            n_layers=num_tx_layers,
            n_heads=num_tx_heads,
            # vocab_size=self._hidden_size,
            max_batch_size=initial_tgt_cache_shape[1],
            max_seq_len=initial_tgt_cache_shape[0],
        )

        state_encoders_linear = OrderedDict()
        state_encoders_time = OrderedDict()
        state_encoders = OrderedDict()  # preserve insertion order in py3.6
        if self.multiple_beliefs:  # multiple belief model
            raise NotImplementedError
        else:  # single belief model
            state_encoders_linear["single_belief"] = nn.Linear(
                tx_input_size, self._hidden_size
            )
            state_encoders_time["single_belief"] = PositionalEncoder(
                self._hidden_size, max_len=self.max_steps
            )
            state_encoders["single_belief"] = TransformerDecoder(state_encoders_params)

        self.state_encoders_linear = nn.ModuleDict(state_encoders_linear)
        self.state_encoders_time = nn.ModuleDict(state_encoders_time)
        self.state_encoders = nn.ModuleDict(state_encoders)

        self.belief_names = list(self.state_encoders.keys())

        get_logger().info(
            "there are {} belief models: {}".format(
                len(self.belief_names), self.belief_names
            )
        )

    def sampler_select(self, keep: list):
        for key, model in self.state_encoders.items():
            if hasattr(model, "sampler_select"):
                model.sampler_select(keep)

    def _recurrent_memory_specification(self):
        return None

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

        # joint_embeds = obs_embeds + prev_actions_embeds
        joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)

        # 2. use Transformers to get single/multiple beliefs
        beliefs_input_dict = {}
        for key, model in self.state_encoders_linear.items():
            beliefs_input_dict[key] = model(joint_embeds)
        for key, model in self.state_encoders_time.items():
            beliefs_input_dict[key] = (
                model(observations[self.time_step_uuid]) + beliefs_input_dict[key]
            )

        beliefs_dict = {}
        for key, model in self.state_encoders.items():
            if joint_embeds.shape[0] > 1 or self.time_step_counter >= self.max_steps:
                self.time_step_counter = 0
            x = beliefs_input_dict[key].permute(1, 0, 2)
            if self.traj_idx_uuid is None:
                mask = None
            elif joint_embeds.shape[0] == 1:
                timesteps = observations[self.time_step_uuid].permute(
                    1, 0
                )  # bs, nsteps
                epi_start = torch.clamp(
                    self.time_step_counter - timesteps, min=0
                ).expand(
                    -1, self.time_step_counter + 1
                )  # bs, 1
                step_range = torch.arange(0, self.time_step_counter + 1).to(
                    device=epi_start.device
                )
                mask = (epi_start <= step_range).unsqueeze(1).unsqueeze(1)
            else:
                traj_idx: torch.Tensor = observations[self.traj_idx_uuid].permute(1, 0)
                mask = traj_idx[:, :, None] == traj_idx[:, None, :]
                mask = torch.tril(mask)
                mask = mask.unsqueeze(1)  # type: ignore
            y = model(x, self.time_step_counter, mask)
            beliefs_dict[key] = y.permute(1, 0, 2)
            if joint_embeds.shape[0] == 1:
                self.time_step_counter += 1

        # 3. fuse beliefs for multiple belief models
        beliefs, task_weights = self.fuse_beliefs(
            beliefs_dict, obs_embeds
        )  # fused beliefs

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

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(beliefs),
            values=self.critic(beliefs),
            extras=extras,
        )

        if self.visualize:
            log_ac_return(actor_critic_output, observations[self.task_id_uuid])

        return actor_critic_output, memory

    @torch.no_grad()
    def compute_total_grad_norm(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm**2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm


class DinoTxGoalEncoder(nn.Module):
    def __init__(
            self,
            observation_spaces: SpaceDict,
            goal_sensor_uuid: str,
            dino_preprocessor_uuid: str,
            depth_preprocessor_uuid: str = None,
            relevant_object_box_uuid: str = None,
            goal_embed_dims: int = 512,
            dino_compressor_hidden_out_dims: Tuple[int, int] = (384, 512),
            combiner_hidden_out_dims: int = 512,
            combiner_layers: int = 3,
            combiner_heads: int = 8,
            concat_depth: bool = False,
            num_depth_channels: int = 1,
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.dino_uuid = dino_preprocessor_uuid
        self.depth_uuid = depth_preprocessor_uuid
        self.relevant_object_box_uuid = relevant_object_box_uuid
        self.goal_embed_dims = goal_embed_dims
        self.dino_hid_out_dims = dino_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.concat_depth = concat_depth
        self.num_depth_channels = num_depth_channels

        if goal_sensor_uuid is not None:
            self.goal_space = observation_spaces.spaces[self.goal_uuid]
            if isinstance(self.goal_space, gym.spaces.Discrete):
                self.embed_goal = nn.Embedding(
                    num_embeddings=self.goal_space.n,
                    embedding_dim=self.goal_embed_dims,
                )
            elif isinstance(self.goal_space, gym.spaces.Box):
                self.embed_goal = nn.Linear(
                    self.goal_space.shape[-1], self.goal_embed_dims
                )
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
            if self.depth_uuid is not None:
                self.depth_pos_encoder = nn.Sequential(
                    PositionalEncoder(32),
                    nn.Linear(32, self.combine_hid_out_dims),
                    nn.LayerNorm(self.combine_hid_out_dims),
                    nn.ReLU(),
                )
                if self.num_depth_channels > 1:
                    self.depth_channel_enc = nn.Embedding(
                        self.num_depth_channels, self.combine_hid_out_dims
                    )

            self.target_obs_combiner = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.combine_hid_out_dims,
                    nhead=combiner_heads,
                    batch_first=True,
                ),
                num_layers=combiner_layers,
            )
            self.fusion_token = nn.Parameter(0.1 * torch.rand(self.goal_embed_dims))

        if relevant_object_box_uuid is not None:
            num_boxes = 1
            num_cameras = 1
            self.len_bounding_boxes = num_boxes * 5 * num_cameras
            self.bbox_pos_encoder = nn.Sequential(
                PositionalEncoder(32),
                nn.Linear(32, self.combine_hid_out_dims),
                nn.LayerNorm(self.combine_hid_out_dims),
                nn.ReLU(),
            )
            self.coord_pos_enc = nn.Embedding(
                self.len_bounding_boxes, self.combine_hid_out_dims
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        return self.combine_hid_out_dims

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

    def encode_depth(self, observations):
        if self.num_depth_channels == 1:
            depth = (
                observations[self.depth_uuid]
                .flatten(start_dim=2)
                .permute(0, 2, 1)
                .squeeze(-1)
            )
            return self.depth_pos_encoder(depth)
        else:
            depth = observations[self.depth_uuid].flatten(start_dim=2).permute(0, 2, 1)
            depth = torch.reshape(depth, (depth.shape[0], -1))
            depth_feat = self.depth_pos_encoder(depth)
            depth_feat = torch.reshape(
                depth_feat,
                (
                    depth.shape[0],
                    -1,
                    self.num_depth_channels,
                    self.combine_hid_out_dims,
                ),
            )
            depth_feat = depth_feat + self.depth_channel_enc(
                torch.tensor(
                    [[i for i in range(self.num_depth_channels)]],
                    device=depth.device,
                ).unsqueeze(1)
            )
            if self.concat_depth:
                return torch.reshape(
                    depth_feat, (depth.shape[0], -1, self.combine_hid_out_dims)
                )
            else:
                return depth_feat.sum(2)

    def distribute_target(self, observations):
        return self.embed_goal(observations[self.goal_uuid])

    def encode_bbox(self, observations):
        best_nav_boxes = observations[self.relevant_object_box_uuid]
        B, T, N = best_nav_boxes.shape
        combined_boxes = best_nav_boxes.reshape(B * T, N)
        pos_encoded_boxes = self.bbox_pos_encoder(combined_boxes)
        pos_encoded_boxes = pos_encoded_boxes + self.coord_pos_enc(
            torch.tensor(
                [[i for i in range((self.len_bounding_boxes))]],
                device=pos_encoded_boxes.device,
            ).tile(B * T, 1)
        )
        return pos_encoded_boxes

    def adapt_input(self, observations):
        observations = {**observations}
        dino = observations[self.dino_uuid]
        if self.goal_uuid is not None:
            goal = observations[self.goal_uuid]
        if self.depth_uuid is not None:
            depth = observations[self.depth_uuid]

        use_agent = False
        nagent = 1

        if len(dino.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = dino.shape[:3]
        else:
            nstep, nsampler = dino.shape[:2]

        observations[self.dino_uuid] = dino.view(-1, *dino.shape[-3:])
        if self.goal_uuid is not None:
            observations[self.goal_uuid] = goal.view(-1)
        if self.depth_uuid is not None:
            observations[self.depth_uuid] = depth.view(-1, *depth.shape[-3:])

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
        visual_feat = (
            self.compress_dino(observations).flatten(start_dim=2).permute(0, 2, 1)
        )
        if self.depth_uuid is not None:
            depth_feat = self.encode_depth(observations)
            if self.concat_depth:
                visual_feat = torch.cat([visual_feat, depth_feat], dim=1)
            else:
                visual_feat = visual_feat + depth_feat
        embs = [
            self.fusion_token.view(1, 1, -1).expand(nstep * nsampler, -1, -1),
            visual_feat,
        ]
        if self.goal_uuid is not None:
            text_feats = self.distribute_target(observations).unsqueeze(1)
            embs.append(text_feats)
        if self.relevant_object_box_uuid is not None:
            pos_encoded_boxes = self.encode_bbox(observations)
            embs.append(pos_encoded_boxes)
        embs = torch.cat(embs, dim=1)
        x = self.target_obs_combiner(embs)
        x = x[:, 0]

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)
