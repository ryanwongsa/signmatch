import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

from gatle_ignite import load_weights
from models.model_utils.hiera.hiera_nodec_mae import mae_hiera_base_16x224

FEATURE_DIM = 768
EMBED_DIM = 512


class HieraEncoder(nn.Module):
    def __init__(self, drop_path_rate):
        super().__init__()
        self.backbone = mae_hiera_base_16x224(
            num_classes=400, pretrained=False, drop_path_rate=drop_path_rate
        )
        self.backbone.patch_embed.proj = nn.Conv3d(
            3, 96, kernel_size=(5, 7, 7), stride=(4, 4, 4), padding=(2, 3, 3)
        )

    def forward(self, windows):
        latent, _ = self.backbone(windows, mask_ratio=0.0)
        return latent


def zeros_skip(conv):
    inflated = nn.Conv3d(
        3, conv.out_channels, kernel_size=(5, 7, 7), stride=(4, 4, 4), padding=(2, 3, 3)
    )
    with torch.no_grad():
        inflated.weight.zero_()
        inflated.weight[:, :, 0::2] = conv.weight
        inflated.bias.copy_(conv.bias)
    return inflated


class ClassPrototypes(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.A = nn.Parameter(torch.empty(num_classes, EMBED_DIM))
        # Off the seeded stream: later modules start as they would in a model without prototypes.
        with torch.random.fork_rng(devices=[]):
            nn.init.xavier_uniform_(self.A)
        with torch.no_grad():
            self.A.copy_(F.normalize(self.A, dim=1))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))


class ClipEncoder(nn.Module):
    def __init__(self, drop_path_rate, num_classes=None):
        super().__init__()
        self.backbone = HieraEncoder(drop_path_rate)
        self.fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(FEATURE_DIM, EMBED_DIM))
        self.confusion = None
        if num_classes is not None:
            self.confusion = ClassPrototypes(num_classes)

    def forward(self, windows):
        return self.backbone(windows)

    def classify(self, clips):
        prototypes = F.normalize(self.confusion.A, dim=1)
        embeddings = F.normalize(self.fc(self.backbone(clips)), dim=1)
        return {
            "logits": embeddings @ prototypes.T,
            "sim_confusion_matrix": prototypes @ prototypes.T,
        }


class ProjectionDict(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(FEATURE_DIM, hidden_size)
        self.projection_dict = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(2 * hidden_size, EMBED_DIM)

    def forward(self, features, lengths):
        sequences = torch.split(self.lin(features), lengths.tolist())
        packed = pack_sequence(sequences, enforce_sorted=False)
        _, (h_n, _) = self.projection_dict(packed)
        last_layer = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return F.normalize(self.proj(last_layer), dim=-1)


class LSTMProjectionDict(nn.Module):
    def __init__(self, hidden_size=EMBED_DIM, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURE_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, features, lengths):
        sequences = torch.split(features, lengths.tolist())
        packed = pack_sequence(sequences, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return F.normalize(h_n[-1], dim=-1)


DICTIONARY_HEADS = {"bilstm": ProjectionDict, "lstm": LSTMProjectionDict}


def dictionary_head_of(state):
    if any(key.startswith("projection_dict.lin.") for key in state):
        return "bilstm"
    return "lstm"


class Model(nn.Module):
    def __init__(self, chunk_size=16, num_classes=None, dictionary_head="bilstm"):
        super().__init__()
        if dictionary_head not in DICTIONARY_HEADS:
            raise ValueError(
                f"dictionary_head {dictionary_head!r}: expected one of {sorted(DICTIONARY_HEADS)}"
            )
        self.chunk_size = chunk_size
        self.backbone = ClipEncoder(drop_path_rate=0.1, num_classes=num_classes)
        self.projection_dict = DICTIONARY_HEADS[dictionary_head]()

    def forward(self, windows=None, lengths=None, clips=None, classify=False):
        """windows: every video's windows concatenated, lengths[i] of them for video i; or clips."""
        if clips is not None and classify:
            return self.backbone.classify(clips)
        if clips is not None:
            return {"clip_embeddings": self.embed_clips(clips)}

        with torch.no_grad():
            features = torch.cat([self.backbone(chunk) for chunk in windows.split(self.chunk_size)])
        outputs = {"dictionary_z_features": self.projection_dict(features, lengths)}

        if self.training and self.backbone.confusion is not None:
            prototypes = F.normalize(self.backbone.confusion.A, dim=1)
            outputs["norm_A"] = prototypes
            outputs["sim_confusion_matrix"] = prototypes @ prototypes.T
        return outputs

    @torch.no_grad()
    def embed_clips(self, clips):
        embeddings = [
            self.backbone.fc(self.backbone(chunk)) for chunk in clips.split(self.chunk_size)
        ]
        return F.normalize(torch.cat(embeddings), dim=-1)


def load_signrep(model, path):
    state = torch.load(path, map_location="cpu")["model"]
    hiera = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    signrep_embed = nn.Conv3d(3, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
    signrep_embed.weight.data = hiera["patch_embed.proj.weight"]
    signrep_embed.bias.data = hiera["patch_embed.proj.bias"]
    inflated = zeros_skip(signrep_embed)
    hiera["patch_embed.proj.weight"] = inflated.weight.data
    hiera["patch_embed.proj.bias"] = inflated.bias.data

    model.backbone.backbone.backbone.load_state_dict(hiera, strict=True)
    dropped = sorted({key.split(".")[0] for key in state if not key.startswith("backbone.")})
    print(f"SignRep: loaded {len(hiera)} backbone tensors, patch embed inflated by zeros_skip")
    print(f"SignRep: dropped {len(state) - len(hiera)} tensors of {', '.join(dropped)}")


def load_stage1(model, path):
    state = torch.load(path, map_location="cpu")
    state = {
        key.removeprefix("backbone."): value
        for key, value in state["model"].items()
        if key.startswith("backbone.")
    }
    load_weights(state, model.backbone, strict=True)
    model.backbone.requires_grad_(False)
