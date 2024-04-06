import prior

prior.load_dataset("procthor-10k")
prior.load_dataset("procthor-10k")

import clip

_, _ = clip.load("RN50", device="cpu")
_, _ = clip.load("RN50", device="cpu")
