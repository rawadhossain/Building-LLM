# Notes Style Guide

Paste this at the start of every new lecture chat, then paste the transcript.

## What these notes are for

They replace rewatching the video. I've already watched each lecture once, so I remember some of it. When I come back months later, the note should bring back everything the instructor taught, including **how he thought about it**: why he introduces an idea, what problem it solves, and how it leads to the next idea.

## How to write them

- Write like notes I'd write for myself: plain, human sentences. Nothing that sounds AI-generated, no filler, no hype.
- **Follow the lecture's order and logic.** Tell it as a chain of "he started here, because of this, which led to that."
- **Explain properly.** Every concept gets a plain explanation, the reasoning behind it, and a small concrete example when that helps. The test: could I understand this note a year from now with the video mostly forgotten?
- **Not too short, not too long.** Skip repetition and padding, but never cut an explanation so far that it stops making sense.
- Keep the instructor's own analogies and examples when they help the idea stick.
- No interview angle unless the instructor brings it up in that lecture.
- **Write about the topic, not the instructor.** No "he says", "he explains", "he admits", "his definition". I know who taught it. State the ideas directly (e.g. "An LLM is a neural network that...", not "His simplest definition is..."). Still follow the lecture's order and keep its analogies and examples, just without attributing them to the speaker.

## Structure of each note

1. Title: `# Lecture NN: <topic>`
2. **If you remember only five things:** a short list at the top.
3. The body, in the lecture's order, with headings for each major idea.
4. Formulas, tensor shapes and dimensions written out clearly where the lecture uses them.
5. A "Code" section, only when the lecture codes something. It links to the matching notebook in the repo with a relative markdown link (e.g. `[1-llmscratch.ipynb](../1-llmscratch.ipynb)`, since notes live in `notes/`) and names the section heading to look for inside the notebook. GitHub can't jump to a section within a notebook, so the link opens the notebook and the named section tells me where to scroll. Skip this section for theory-only lectures.
6. A short "what's next" line.
7. No "transcript fixes" section. Fix caption errors silently.

## Diagrams

- Use Mermaid blocks (GitHub renders them) when an arrow diagram or flow really helps, such as architecture, data flow or step sequences. Don't force one into every note.
- For anything Mermaid can't draw well, say so and I'll add an image myself.

## Gaps the transcript can't capture

- If the instructor relies on a slide, drawing or code on screen that the transcript doesn't capture, leave a marker: `[check video ~mm:ss: what to look at]` (or without a time if the transcript has no timestamps).
- Auto-captions garble technical terms. Fix them silently from context and don't mention the fixes in the note. If a spot is genuinely ambiguous, mention it in the chat reply, not inside the note.

## Screenshots

When I attach a screenshot from the video, it's because I want it in the note. I won't attach everything, so if I send one, use it.

- Save it in `notes/images/`, never loose in `notes/`. Pasted screenshots arrive as `image.png`, so rename them right away to avoid collisions.
- Name format: `NN-short-topic.png`, where `NN` is the lecture number and the rest describes what the image shows (e.g. `03-pretraining-finetuning-schematic.png`). If a lecture needs two images of the same thing, add a suffix like `-2`.
- Embed it with a relative link and real alt text, not "alt text": `![what it shows](images/03-pretraining-finetuning-schematic.png)`.
- Place it under the matching `[check video ~mm:ss: ...]` marker. Match the image to the right marker by its content, and if the timestamp in the screenshot or the existing marker looks off, tell me in the chat reply.
- Keep the marker next to the image so I can still jump to that point in the video. If the image fully replaces what the marker pointed to, the marker can stay as the timestamp reference.
- Screenshots often contain the instructor's handwriting or slide text. Describe the key labels in the note's own text too when that helps, so the note still makes sense if the image is skipped.

## Files

- One markdown file per lecture, numbered: `01-series-introduction.md`, `08-gpt-tokenizer-bpe.md`, etc.
- I commit and push myself, so just give me the file.
