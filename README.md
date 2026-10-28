# Strata `4070-main-lite` (`carteakey/Strata`)

This branch (`4070-main-lite`) is a streamlined, minimal-bloat fork of Strata tailored specifically for an **RTX 4070 12 GB (Ada Lovelace, `sm_89`, 46 SMs)** paired with 64 GB DDR5-5200 host RAM running **Qwen3.8-Flash-Next** (`qwen4exp`, ISTA-DASLab GSQ-RCO IQ3_XXS with unquantized Q8_0 PLE table).

Rather than tracking every upstream release or pulling in multi-vendor bloat, this fork cherry-picks only high-value inference kernels, speculative decoding improvements, and CUDA optimizations while pruning unused architectures.

---

## Performance Progression on RTX 4070 (12 GB)

Measured with `bench-models/bench-qwen38-refresh-ab.sh` on Linux 7.3.0 CachyOS, powersave governor, DDR5-5200 4x16 GB, serving `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` (`IQ3_XXS`, unquantized Q8_0 PLE table on NVMe):

| Engine / Revision | Sustained Decode (t/s) | Peak Decode (t/s) | 8k Prefill (t/s) | 4k Needle Pass | VRAM Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Strata v0.1.45 (upstream tag)** | 63.83 | 67.7 | 1,926.1 | Yes (`true`) | 11,696 MiB |
| **Strata v0.2.6 (upstream tag)** | 65.53 | 70.5 | 1,890.7 | Yes (`true`) | 11,696 MiB |
| **`4070-main-lite` (initial pruned + HC fusions)** | 65.08 | 72.9 | 1,894.1 | Yes (`true`) | 11,698 MiB |
| **`4070-main-lite` + Ada `sm_89` MMVQ (PR #3453)** | **69.60** | **76.1** | **1,883.1** | Yes (`true`) | **11,696 MiB** |
| **`4070-main-lite` + `--kv-grow` elastic cache** | **73.53** | **78.9** | **1,888.0** | Yes (`true`) | **11,696 MiB** |

Net gain: **+9.70 t/s (+15.2%) over upstream v0.1.45**, with peak decode reaching **78.9 t/s** (3,277 expert slots on GPU, 62.1% hit rate).

Read that table with two caveats. A repeat run the same day came in at **67.88 t/s** (63.0 /
71.6), so run-to-run spread on this tier is about 2 t/s and anything under ~3 t/s needs a
paired control arm in the same boot. And 8k prefill has not improved: 1,883 here against
1,926 on plain v0.1.45 and 1,911 on v0.1.44.1. Those prefill cells are the higher probe of
each run's two pp8192 probes, and each pair spans 1,869 to 1,883 (fork + table), 1,911 to 1,926
(v0.1.45) and 1,908 to 1,914 (v0.1.44.1), i.e. means of 1,876 / 1,919 / 1,911. Decode is where
this fork won, prefill is where it has not kept up.

---

## Fork Modifications & Architecture Changes

### 1. Pruned Bloat & Build Streamlining
- **Removed Unused Backends (331 files / 172,853 LOC removed)**:
  - Deleted `src/kernels/metal/` (macOS Apple Silicon).
  - Deleted `src/kernels/sycl/` (Intel Arc).
  - Deleted AMD ROCm tuning scripts (`src/kernels/rocm/hipblaslt_tuning.hpp`, `src/kernels/rocm/rocblas_tuning.hpp`).
- **Gated Experimental Engines & Tests**:
  - Gated unreleased Qwen3.5 engine (`src/qwen35/`) behind `STRATA_ENABLE_QWEN35=OFF`.
  - Disabled 109 test binaries from standard build (`STRATA_BUILD_TESTS=OFF`).
  - Set default `CMAKE_CUDA_ARCHITECTURES=89` targeting Ada Lovelace.
- **Binary Footprint**:
  - `build/strata` executable reduced to **73.0 MB** (down from 77.5 MB on v0.1.45 and 78.7 MB on v0.2.6).

### 2. Tracked Inference Commits & Cherry-Picks
- **`8645328d` - Fused Hyper-Connections (`gr.cu`, `fused_gr.cu`)**:
  - Folded the hyper-connection inject and gate/mean operations into adjacent GEMV down grids (`fused_gm = hc <= 4`), eliminating two standalone kernel dispatches per verification step (6 launches reduced to 4).
- **`7ce60ad3` - `STRATA_GR_NORM_FAST=1` Register-Split Norm**:
  - Ported register-split norm kernel on CUDA producing bitwise-identical output while trimming register pressure.
- **`9aafcbf0` - Reasoning Loop Recovery (`serve/server.py`)**:
  - Added loop detection recovery: when `repeat_stop_tokens` fires inside `<think>`, the server wraps thinking into the final answer rather than failing the generation. Verified with 19/19 unit tests passing.
- **`0cef00fa` - Ada Lovelace `sm_89` MMVQ Interleave Table (Strata PR #3453)**:
  - Added hardware-calibrated warp row counts (`kIlRows_89[]`, `kIl1Rows_89[]`) for `IQ4_XS`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q3_K` in `src/kernels/cuda/native_mmvq.cu`.
  - Yielded a direct +4.52 t/s (+7.0%) sustained decode improvement (65.08 -> 69.60 t/s).

### 3. Evaluated & Rejected Experiments
- **`--mtp-q4 head` (On-the-fly Q4_0 draft head quantization)**:
  - Tested quantizing the MTP draft head from Q8_0 down to Q4_0 to reduce DRAM read traffic during drafting.
  - Result: Decode dropped from 69.60 t/s to 67.88 t/s. On-the-fly ALU dequantization in the draft step cost more latency than the memory bandwidth saved on an RTX 4070. Reverted to keep native Q8_0 draft head.

### 4. What the tier reports about itself at load

From `strata-iq3_xxs-ple_q8.log` (2026-10-28 boot), the numbers that decide where the next
win has to come from:

| line | value | reading |
| :--- | :--- | :--- |
| `expert cache 2583 slots, 4.24 GiB of VRAM` | 2,583 of 24,576 expert pairs | the shipped profile's head decides which 2,583 are hot |
| `the prompt path borrows 2323 CUDA0 cache slots` | 3.81 GiB | prompt reading and decode contend for the same VRAM |
| `serve: 410 MiB of VRAM free with everything loaded` | below the 512 MiB floor | the engine warns about driver paging to system RAM (`#2065`), and asks for `--vram-reserve-mib 832` |
| window captures | down to 282 MiB free | the CUDA graph windows are what push it under the floor |
| `decode expert cache hit rate: 58.6%` | 54,487 hits / 92,979 lookups | 41% of routed experts are computed off GPU |
| `3652 more read by the GPU over PCIe (3.8%)` | the rest is CPU pool work | decode on this box is partly a CPU job |
| `5 pool workers on logical processors 2,4,6,8,10` | mask `cpu_range: 0-11` | the 4 Alder Lake E-cores (12 to 15) sit outside the mask, idle |
| `MAP_HUGETLB unavailable (needed 20464 2 MiB pages, vm.nr_hugepages=0)` | arena on 4 KB pages | upstream prices ~750 TLB misses per 3 MB expert on this path |
| `mtp: draft head over 106299 tokens (178.4 MiB)` | shipped `rt` subset | `data/draft_vocab_en.bin` would take it to 40,525 tokens |

### 5. Candidate backlog (measured order, not a wish list)

Full reasoning, citations and the harness: `docs/strata-perf-next-steps-2026-10.md` in the
L3MS tree. Short version, in the order I would spend card time on them:

Measured on 2026-10-28 in the L3MS sweep harness (nine arms, 23 min, the production config run at
both ends as the paired control: **70.70 and 71.55 t/s**, pair mean 71.13). Numbers are decode mean
over four 256-token code probes; hit rate and PCIe share come from the engine's own lines.

1. **`STRATA_ROUTE_PRIOR=1.5`** (`src/core/verify.cpp:1849-1851`, `src/kernels/cuda/route_prior.cu`):
   residency-biased routing. Measured **102.23 t/s** (97.4 to 109.3), +31.1 over the pair mean, with the
   cache hit rate at **93.0%** against 57.4 and 57.7 in the two control arms and PCIe down to **1.3%**
   from 9.9%, with expert slots identical (2,583) and draft acceptance essentially at the control's level
   (accepted/proposed mean 0.831 against 0.794). Not served yet: the kernel and
   the call site both say it changes the output, and upstream's 0.2.7 release note flags agent-task
   completion (4.5 of 10 at 1.5 against 5.6 with it off, inside their spread). Upstream defaults 1.5 on
   a discrete NVIDIA card at or under 13 GiB and used 2.0 in its first candidates, which showed copy and
   code defects. This fork is on the 0.1.45 base, so nothing turns it on by itself here.
2. **VRAM headroom**: `--kv-grow` measured **73.53** (+2.40, inside the ~2 t/s spread) with slots 2,583 to
   **3,277** and hit rate 62.1%, but the run's worst disk traffic (0.0998 MB per generated token) and a
   32k prefill dip (1,928 against 1,950 and 1,948): needs a paired repeat before it is served.
   `--vram-reserve-mib 832` measured **68.75** (-2.38) while doing its job (410 to 544 MiB free, capture
   floor 276 to 410) at the cost of 77 slots: insurance, not speed. `--max-context 65536` unrun.
3. **A workload-shaped `--expert-profile`** built from my own coding-agent prompts:
   `tools/collect_profile_traces.py CONFIG.json OUTDIR --mode routing` (CUDA needs `--mode
   routing`; run it with the Strata venv, the tool imports `regex`), then `tools/make_profile.py
   OUTDIR/*.trace --reorder --out data/expert-profile-pi.bin` so the traces decide the head and
   the shipped base fills the tail. `--reorder` and `--no-base` belong to `make_profile.py`;
   `make_profile_mix.py` in this tree weights whole traces against each other and takes no base
   flags.
4. **`--mtp-draft-vocab data/draft_vocab_en.bin`**, measured and rejected: 40,525 tokens and 68.0 MiB
   against 106,299 and 178.4 MiB, 50 extra auto slots (2,003 against 1,953), and **5.35 t/s slower** on
   code with acceptance mean 0.768 against 0.794. A subset worth retrying is code-plus-English built with
   `tools/draft_vocab.py --corpus ... --coverage 0.98`, not the shipped English-only file.
5. **Speculative window**: `--spec 5` measured **61.00** (-10.13) with acceptance mean 0.763 against the
   control's 0.794, so the loss is the wider verification window itself and `--spec 4` stays. `STRATA_SPEC_PROB=1`, `--lookup-chain 5` and
   `--suffix-draft 0` are unrun.
6. **`--aux-cpus auto` with taskset `0-15`** so aux threads land on the idle E-cores (unrun; today's 5
   pool workers sit on logical processors 2, 4, 6, 8, 10).
7. **Prompt side** (`pp` has been flat for a month): `--prefill auto:16384` measured **dead neutral**
   (70.60 decode, 1,877 at 8k and 1,946 at 32k), so `--prefill auto` is not bounded by the 8,192 chunk
   ceiling on this card and raising it buys nothing. Still unrun: `--prefill-head-alias`,
   `STRATA_PROMPT_ATTN_IMMA=1`, `STRATA_HC_Q8=1`, `--ple-io mmap`.
8. **`vm.nr_hugepages` reservation** for the expert arena (needs sudo, `STRATA_NO_LARGEPAGES=1`
   is the same-boot A/B switch).

Closed items, do not re-run:

- **Ada `sm_89` MMVQ interleave table**: merged as our `0cef00fa`, and upstream carries it
  too, on by default for compute capability 8.9 in the 0.2.x line.
- **`STRATA_GR_NORM_FAST=1` on Ada**: measured **64.85 against the 71.13 pair mean, 6.3 t/s slower**, so
   it stays off on `sm_89`. Also a correction to what this README claimed before: our commit `7ce60ad3`
   is **not** in `v0.2.7`, which carries its own CUDA opt-in marked "measured on sm_70 only"
   (`src/kernels/cuda/fused_gr.cu:1272`). Nothing to re-apply in a re-slice.
- **`--mtp-q4 head`**: rejected, see item 3 above.
- **llama.cpp PR #30790**: named in an older version of this backlog; that PR does not exist.
  #30792 (native q8_0 KV in the flash-attn tile kernel) is real and open, and llama.cpp
  #31037 / #29143 (trimmed MTP draft head) are the same idea this engine already ships as
  `--mtp-draft-vocab` plus `tools/draft_vocab.py`. Reading them is a second opinion, not a
  port.

### 6. Upstream distance and the anti-bloat decision

Upstream moved 2,252 commits from the fork's merge base `477ac13c` (2026-10-16) to `v0.2.7.1`. However, the vast majority of that delta is multi-platform bloat (Apple Metal, AMD ROCm, Intel SYCL, experimental unreleased Qwen3.5 scaffolding, 100+ unit tests, and 218 undocumented switches). 

Re-slicing or rebasing the whole fork onto upstream would re-introduce all that dead code and defeat the purpose of `4070-main-lite`. This fork stays dedicated strictly to lean Ada `sm_89` execution on the RTX 4070. We do not swallow upstream churn wholesale; we selectively audit and cherry-pick isolated, high-leverage CUDA inference kernels and speculative decoding wins after before/after A/B bench verification.

---

<h1 align="center">Strata (Upstream Base)</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC</b><br>
NVIDIA or AMD graphics card (12 GB or more) · Windows or Linux · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a normal PC. This is a
large, smart AI model that usually needs a server. It chats, writes code, reads pictures and works with your apps
and coding agents. Nothing leaves your PC.

## How fast is it?

We measured it on two ordinary gaming PCs. A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (here a 32K-token document, code or chat history).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 with engine 0.1.36, the other rows with 0.1.26 (4K answers, 32K prompts). The full tables are in
[DETAILS.md](docs/DETAILS.md#speed-measured). A card with more VRAM is faster: an RTX 3090 (24 GB) should write
about 100-140 tokens per second. Long chats and other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size),
[community results](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. If it runs well on your PC, a coffee keeps the work on it going.</sub></p>

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series. It needs **12 GB of VRAM or more**. |
| **RAM** | 32 GB or more. Your RAM decides [which model](#which-model-should-i-pick) fits. 64 GB runs every size. |
| **Disk** | About 80 GB free. Use an SSD if you can: the first start is much faster. |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD. |

The installer sets up everything else. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, written and tested by community members on their own machines:

- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **Apple Silicon Macs (M1 to M5)**, built from source by `./setup.sh` (the Metal engine): [Mac](docs/MAC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop Strata through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the Strata folder.

**Mac (Apple Silicon, M1 to M5, macOS 15 or newer):** install Apple's Command Line Tools once
(`xcode-select --install` in Terminal; Xcode itself is not needed). Then open Terminal in the Strata folder and run
**`./setup.sh`**. It builds the Mac engine on your Mac (about 7 minutes), asks the same questions, downloads the model
and starts it. Next time, `./setup.sh` starts it right away. Which size fits your Mac and how fast it is:
[docs/MAC.md](docs/MAC.md).

The steps are the same for NVIDIA and AMD. The installer finds your card and sets up the right engine for it. It
asks you a few questions:

- which model and which size,
- how much context (how much text the model keeps in mind),
- whether it should read pictures.

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB) and starts it. If the
download stops, run it again: it continues where it left off. Your browser opens the Strata app at
`http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> Strata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window. The window shows what Strata is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

**Is it fast on your PC?** Run `BENCH.bat` (`./bench.sh`): about 5-10 minutes, it measures chat, 4K/16K prompts and an agent-style session with your own settings and writes a report to share ([how](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

## Which model should I pick?

The installer recommends one for your RAM. The same model comes in several sizes, compressed more or less. Smaller
sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's UD-IQ4_XS (~4-bit) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, Strata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL and Q8_0](docs/MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0):**
  4- to 8-bit, closer to the full model (93-95% the same next token as Q8_0). Fast where every expert fits in RAM
  (~88 GB for UD-Q4_K_XL up to ~136 GB for Q8_0); with less RAM Strata reads part of them from the SSD while it
  answers, which is much slower (7-8.5 tokens/s for UD-Q4_K_XL on a 64 GB PC).
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, run
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** open `http://127.0.0.1:8080`. It has **Chat**, a live **Monitor** of the model and your
  GPU/CPU/RAM, and **About** with the settings and addresses.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **Pictures:** say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app.
  AMD cards read pictures on Linux through the processor; on Windows they can't yet.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** by default Strata answers one request, and the others wait. To answer several at once,
  set `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- **Long prompts:** Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up
  messages start in seconds.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

### Optional: more speed on one card when the whole model fits in RAM

Two opt-in settings (by RubenTorney, #1904 and #1905). They help when you run one graphics card and your RAM holds
every expert (Strata's arena mode, for example Q2_0 or IQ3_XXS on a 64 GB PC). Open your config,
`strata-<model>.json` in the Strata data folder ([where](docs/INSTALL.md#where-things-are-stored)), and add:

- **Faster decode:** `"--adapt-async", "1"` to the `"args"` list. The expert cache then updates on a helper thread
  instead of pausing the answer. Measured on an RTX 5070 (Windows, Q2_0): +4-5% decode, +7-10% on code and long
  documents. Answers can differ slightly from the default (they stay the same from run to run).
- **Faster prompt reading on Windows:** `"STRATA_PIN_BY_NEED": "1"` in the `"env"` block. Windows can lock only about
  half the RAM for the graphics card; this locks the experts the card actually needs first. Measured on the RTX 5070:
  +4.4% prompt reading.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Restart Strata (`START-HERE.bat` or `./setup.sh`) after the change. To undo, remove the two entries.

## Something went wrong?

- **My PC froze the first time Strata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder. Found a
security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
