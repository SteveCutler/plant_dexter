# Pipeline and local artifacts

Run scripts from the repository root. This guide describes the cleaned source and intended handoffs; a full run has not been reproduced because the original artifacts and environment are unavailable.

## Environment and inputs

Imports require PyTorch, OpenCLIP (`open_clip_torch`), pandas, NumPy, Pillow, tqdm, Hugging Face Datasets, Requests, and scikit-learn. OpenCLIP uses torchvision for image transforms. Core ML conversion additionally requires `coremltools`. Restore the original environment if available or verify and record a new compatible environment; no historical package versions are claimed here.

Place the Ontario species CSV at `data/OntarioPlants.csv`. The original Excel-to-CSV preparation is not scripted. Training photos default to `data/PlantImages/`; set `PLANTDEXTER_IMAGE_ROOT` consistently before downloading, counting, topping up, or generating pairs to use another location. Other script path settings remain in their configuration blocks.

`data/`, `embeddings/`, `checkpoints/`, and `artifacts/` are ignored. Raw species tables, downloaded images, split/pair CSVs, trained weights, reference vectors, and exported Core ML packages are intentionally not committed. Neither is a Swift/Xcode app.

## 1. Prepare the dataset

| Run in this order | Reads | Writes or downloads |
| --- | --- | --- |
| [`preprocessing/process_ontarioplants.py`](../preprocessing/process_ontarioplants.py) | `data/OntarioPlants.csv` | `data/ontario_native_filtered.csv` |
| [`preprocessing/data_search.py`](../preprocessing/data_search.py) | Filtered CSV and streamed BioTrove records | `data/ontario_biotrove_matches.csv`, `data/ontario_biotrove_overlap.csv` |
| [`preprocessing/image_pull.py`](../preprocessing/image_pull.py) | Matches CSV | Up to 200 selected image URLs per species into the image root |
| [`preprocessing/check_image_counts.py`](../preprocessing/check_image_counts.py) | Image root | `data/low_count_species.csv` |
| [`preprocessing/scrape_images.py`](../preprocessing/scrape_images.py) | Low-count CSV and iNaturalist observations | Additional images toward 100 per species |
| [`modeling/generate_image_text_pairs.py`](../modeling/generate_image_text_pairs.py) | Image root | `data/plantdex_clip_pairs.csv` |
| [`preprocessing/data_split.py`](../preprocessing/data_split.py) | Full pairs CSV | `data/train_pairs.csv`, `data/val_pairs.csv` |

Each entry is a standalone script, for example `python preprocessing/process_ontarioplants.py`. The plant filter still selects vascular plants with rank strings matching `S4`, `S5`, or `SNA`; the native-only condition remains disabled. Dataset selection logic was not changed in this cleanup.

Pairs now contain `image_path`, `text_prompt`, and `scientificName`. The prompt still uses one of the original three forms, such as `a photo of Acer rubrum`. Source image paths are carried into the embedding mapping. No numeric taxon identifier is present in the original matched-CSV schema, so none is invented.

## 2. Fine-tune and choose a checkpoint

```bash
python preprocessing/data_split.py
python modeling/biotrove_fine_tune_sc.py
```

The split remains a 90/10 random row split with seed 1337. Training now reads **only `data/train_pairs.csv`**; the original prototype read the complete dataset. This preserves the intended separation for new runs. Existing checkpoints retain their original training history, and duplicate/related observations may still cross the row split.

The trainer keeps its full-model symmetric image/text contrastive loss, AdamW optimizer, batch size 16, two configured epochs, and learning rate `5e-6`. It writes model state dictionaries to `checkpoints/` every 500 steps, at each completed epoch, and to `clip_finetune_final.pt` on completion. Importing the trainer no longer starts a run. The historical development timing is approximately 12.5 hours.

The embedding and evaluation scripts retain the historical default **`checkpoints/clip_finetune_step41000.pt`**. Select an available checkpoint in their configuration blocks before running them; a short/new run might not produce step 41,000. Use the same checkpoint for reference embeddings and export. The TorchScript CLI also accepts `--checkpoint`.

| Script | Input | Output |
| --- | --- | --- |
| [`modeling/eval_clip.py`](../modeling/eval_clip.py) | `data/val_pairs.csv` and selected checkpoint | Image-text row-retrieval metrics on stdout |
| [`modeling/eval_test.py`](../modeling/eval_test.py) | Selected checkpoint and `tests/cappadocian_navelwort.jpg` | Three scores among ten manually selected labels |

Both use OpenCLIP's inference transform, but their scoring logic is unchanged. `eval_clip.py` counts only the matching CSV row as correct, even when another row describes the same species. It falls back to the base model when the checkpoint is missing and uses non-strict weight loading. These behaviors need review before interpreting its output as species accuracy. `eval_test.py` is a manual spot check, not a test suite.

## 3. Generate embeddings and preserve row identity

```bash
python modeling/create_embeddings.py
```

Inputs are the full pairs CSV, `data/ontario_biotrove_matches.csv`, and the selected checkpoint. The reference catalog still covers the full dataset, as in the prototype; it must not be used unchanged as a held-out validation retrieval index.

Outputs under `embeddings/` are:

| File | Contents |
| --- | --- |
| `plant_dexter_embeds.pt` | `image_embeddings` and `text_embeddings`, normalized before saving |
| `plant_dexter_embedding_rows.jsonl` | One record per row in both embedding matrices |
| `plant_dexter_metadata.json` | Species-level reference metadata keyed by scientific name |

A mapping record looks like:

```json
{"embedding_index": 0, "scientificName": "Acer rubrum", "image_path": "data/PlantImages/acer_rubrum/example.jpg", "metadata_key": "Acer rubrum"}
```

The invariant is **matrix row N → JSONL record N**, with zero-based indices. Each dataset sample carries its scientific name and image path through the same batch as its image and text. Mapping records are appended in that batch order, after encoding, even if the species has no extra metadata (`metadata_key` is then null). Unreadable images are skipped before batching, so they create neither vectors nor mapping records. A length check runs before saving.

Legacy two-column pair CSVs remain supported by the original prompt-name parsing. New pair CSVs carry the scientific name explicitly. The tensor-file keys remain unchanged, and the species dictionary is retained for descriptions and display. Keep the JSONL sidecar with every representation of these matrices; do not reconstruct row identities from the source CSV or the species dictionary.

Embedding generation uses OpenCLIP's validation transform, batch size 32, six data-loader workers, and the existing autocast path. The historical optimized workflow took approximately 3.5 hours after an initial projection of roughly 10 hours; no new timing was measured.

## 4. Attach descriptions and reference images

| Run in this order | Reads | Writes |
| --- | --- | --- |
| [`modeling/desc_scrape.py`](../modeling/desc_scrape.py) | Species metadata and Wikipedia summaries | `embeddings/plant_dexter_metadata_wiki.json` |
| [`modeling/get_display_images.py`](../modeling/get_display_images.py) | Enriched metadata and Wikimedia/iNaturalist images | JPEGs in `data/plant_images/` and `embeddings/plant_dexter_metadata_with_local_images.json` |
| [`deployment/ios/compress_ref_images.py`](../deployment/ios/compress_ref_images.py) | `data/plant_images/` | WebP files in `data/plant_images_webp/` |

Compression now reads the image collector's output directory. It does not rewrite the metadata's JPEG paths; choosing WebP for the app still requires that handoff to be updated. Source attribution should travel with any assets that are later bundled.

## 5. Compare embedding storage formats

| Script in [`deployment/ios/`](../deployment/ios/) | Reads | Writes under `embeddings/` |
| --- | --- | --- |
| `quantize_embeds_fp16.py` | Original `.pt` embeddings | `plant_dexter_embeds_fp16.pt` |
| `quantize_embeds_int8.py` | Original `.pt` embeddings | `plant_dexter_embeds_int8.pt` |
| `quantize_embeds_fp16_npz.py` | FP16 `.pt` file | `plant_dexter_embeds_fp16.npz` |

FP16 and INT8 are alternative conversions; NPZ follows the FP16 conversion. The algorithms are unchanged: INT8 stores values and a scale per tensor, and NPZ compresses NumPy arrays. All retain row order, so each uses the original `plant_dexter_embedding_rows.jsonl` alongside it. These scripts do not copy or embed the mapping; bundle it explicitly with the chosen representation. Regenerate the mapping with its tensors if rebuilding references.

## 6. Export the encoder for Core ML

After selecting the same checkpoint used for reference generation:

```bash
python deployment/ios/convert_to_torchscript.py --checkpoint checkpoints/clip_finetune_step41000.pt
python deployment/ios/convert_ts_to_coreml.py
```

The first script accepts raw state dictionaries and dictionaries containing `state_dict`. It **strictly applies the weights before extracting the visual encoder**, then exports normalized image features. Outputs default to:

- `artifacts/ios/plantdex_image_encoder_traced.pt`
- `artifacts/ios/plantdex_image_encoder_traced.preprocessing.json`
- `artifacts/ios/PlantDex_ImageEncoder_FP16.mlpackage`

Export accepts `--output`; conversion accepts `--input` and `--output`. Both have `main()` guards. The preprocessing sidecar records the loaded model's preprocessing configuration, transform description, input layout, checkpoint path, and Torch/OpenCLIP versions. Keep it with the exported model. The prototype's `check_trace=False` workaround remains, so successful tracing alone is not evidence of numerical agreement.

### Exact input preprocessing

The Core ML input `input_image` is a **float32 `MLMultiArray` / tensor with shape `[1, 3, 224, 224]` in NCHW order**. It does not accept raw image pixels. Normalization is outside the exported graph, as are resize and crop.

For the published [BioTrove OpenCLIP configuration](https://huggingface.co/BGLab/BioTrove-CLIP/blob/main/open_clip_config.json), the inference input is prepared as follows:

1. Convert the decoded image to RGB, matching the Python embedding loader.
2. Resize the shorter side to 224 pixels, preserving aspect ratio, using bicubic interpolation.
3. Take the central 224 × 224 crop.
4. Convert RGB bytes to float32 values in `[0, 1]` and arrange channels as CHW.
5. Normalize each channel with `(value - mean) / std`, using mean `[0.48145466, 0.4578275, 0.40821073]` and standard deviation `[0.26862954, 0.26130258, 0.27577711]`.
6. Add the batch dimension to obtain `[1, 3, 224, 224]`.

Resize/crop behavior comes from [OpenCLIP's inference transform](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transform.py). Training keeps its original augmentation transform. Use the same library environment for embedding generation and export; the generated sidecar records the actual loaded configuration. If it differs from the published values above, reconcile that difference before preparing app inputs.

The [Python reference helper](../deployment/ios/preprocess_image.py) calls OpenCLIP's inference transform directly:

```bash
python deployment/ios/preprocess_image.py path/to/photo.jpg --output artifacts/ios/input_image.npy
```

It may download the base model through the OpenCLIP factory. It does not approximate the transform or perform a Core ML prediction. The consuming iOS application must reproduce the transform, including image orientation and resize/crop rounding. Python currently does not apply an extra EXIF-orientation correction. Compare Python and app inputs/outputs on actual images before claiming deployment parity.

## Verification limits and deferred issues

No original data, fine-tuned checkpoint, reference vectors, Core ML package, or iOS source is available in this checkout. Source checks cannot validate historical accuracy, conversion compatibility, device latency, or memory use.

The cleanup leaves these behavior changes for later: species-level scoring and observation-aware splitting; normalization of species keys in the overlap report and metadata joins; black-image substitution for corrupt training/evaluation images; inconsistent failure values in Wikipedia descriptions; JPEG-to-WebP metadata updates; and the trace-check workaround. Several older data/metadata/storage scripts still execute at import, so run them as scripts after reviewing their inputs. Their algorithms and the original training architecture have been preserved.
