# License check — gemma4-spark derivative checkpoint (2026-09-21)

Internal provenance note; not part of the upload set.

Question: may the untied-lm_head NVFP4 derivative of
`nvidia/Gemma-4-26B-A4B-NVFP4` be redistributed on HF?

Sources:

- Publication brief (2026-09-21): the Gemma 4 base license is Apache
  License 2.0, confirmed at ai.google.dev.
- HF metadata on both upstream repos reads `apache-2.0`; the `google/`
  repos additionally carry the Gemma Terms of Use and a prohibited-use
  policy; the NVIDIA card states commercial/non-commercial use is OK.
- This derivative is built from `nvidia/Gemma-4-26B-A4B-NVFP4`
  (NVIDIA's quantization of `google/gemma-4-26B-A4B-it`), with
  `tie_word_embeddings=false` and a separately NVFP4-quantized
  `lm_head.weight` produced by `untie-lmhead-fp8.py`.

Conclusion: redistribution under Apache License 2.0. What ships:

- `LICENSE` — Apache License 2.0 verbatim
- `NOTICE` — derivative statement naming Google (base model) and
  NVIDIA (NVFP4 quantization), and the Gemma ToU pointer
- model card `license: apache-2.0` metadata

Caveat kept on record: the Gemma Terms of Use / prohibited-use policy
attach to the model itself and ride along regardless of the code
license; the card repeats this so downstream users see it.
