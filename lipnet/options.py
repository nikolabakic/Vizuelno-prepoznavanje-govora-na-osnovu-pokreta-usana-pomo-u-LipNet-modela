"""Shared baseline parameters, close to VIPL ``options.py``.

Phase 5 will consume these values. Phases 0-4 only display or smoke-test them.
"""

random_seed = 0
vid_padding = None  # AI-SPEAK is padded dynamically by variable_length_collate
txt_padding = None
batch_size = 2
base_lr = 2e-5
num_workers = 2
max_epoch = 10000
display = 10
test_step = 1000
save_prefix = "weights/LipNet_serbian"
is_optimize = True

# Exact VIPL unseen-speaker checkpoint pinned with the source commit.
vipl_commit = "40209e09c49553c00c25c7d41faa3706aea3c625"
vipl_checkpoint_name = (
    "LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_"
    "cer_0.06796452465503355.pt"
)
vipl_checkpoint_url = (
    "https://raw.githubusercontent.com/VIPL-Audio-Visual-Speech-Understanding/"
    f"LipNet-PyTorch/{vipl_commit}/pretrain/{vipl_checkpoint_name}"
)
