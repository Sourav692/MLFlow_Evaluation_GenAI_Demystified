Good. Now let me build the full explanation with carefully sequenced diagrams.

This is a genuinely important concept to understand before learning MLflow — it changes how you think about evaluation entirely. Let me walk through it systematically.

---

## Why Evaluating LLMs and Agents is Fundamentally Different

Traditional software has deterministic, binary outputs — a function either returns the right answer or it doesn't. LLMs break every assumption that classical testing and evaluation was built on. Here's why, piece by piece.

---

### Problem 1: The Same Input Produces Different Outputs Every Time

A traditional unit test passes or fails permanently. With an LLM, the same exact prompt can generate a brilliant answer, a mediocre one, or a wrong one — on consecutive calls. This is non-determinism caused by temperature sampling, and it means a single evaluation run tells you almost nothing on its own.

![Determinism Comparison](Image1.jpg)

---


### Problem 2: Classical Metrics Completely Break Down

For decades, NLP used metrics like BLEU and ROUGE — they count word overlaps between the model's output and a "reference" answer. These worked when there was one correct translation or one correct summary. They collapse entirely for open-ended generation.The core problem: BLEU rewards copying the reference's words, not getting the answer right. An answer that is semantically identical but uses different vocabulary gets penalised. An answer that is factually wrong but uses similar words gets rewarded. These metrics have the opposite incentive structure you need.

---

### Problem 3: There Is No Single "Right Answer"

For open-ended generation, correctness isn't binary — it exists on a spectrum and depends on context.---

### Problem 4: Agents Add Three More Failure Dimensions

A standalone LLM is hard enough to evaluate. An agent — one that reasons in multiple steps, calls tools, retrieves documents, and produces a final answer — adds entirely new ways to fail.A correct final answer can hide failures at layers 1 and 2. The agent might have retrieved completely wrong documents, made 6 redundant tool calls, and still guessed the right answer from its training data — your evaluation would show a green score while the system is fundamentally broken.

---

### Problem 5: The Evaluation Triangle — You Can't Have Everything

Every evaluation decision is a tradeoff across three [axes.No](http://axes.No) single approach wins — that's why MLflow's evaluation stack uses all three in combination: code-based scorers for cheap deterministic checks, LLM judges for semantic quality at scale, and human feedback for ground truth calibration.

---

### The Solution: A Three-Paradigm Stack---

### The Summary — Five Reasons It's Different

| Classical software | LLMs and agents |
| --- | --- |
| Deterministic output | Non-deterministic — same input, different outputs |
| Binary pass/fail | Spectrum scoring — partially right is a real state |
| Word-overlap metrics work | Semantic quality requires judgment, not counting |
| Single failure mode to test | Three failure layers in agents (retrieval, reasoning, generation) |
| Unit test once, it holds | Must evaluate statistically, at scale, continuously |

The reason MLflow's evaluation framework exists exactly as it does — with LLM judges, code-based scorers, human feedback, sampling configs, and production monitoring — is because each of these five differences demands a specific tool. Module 1 sets the conceptual foundation; everything in Modules 2–8 is the concrete engineering response to these problems.