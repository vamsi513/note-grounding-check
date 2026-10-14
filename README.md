# note-grounding-check

A small tool that checks whether each sentence in a generated clinical style note is supported
by the source conversation, and flags the claims that are not. This is a reliability and
verification study. **It is not clinical software.** It has no clinical validity, it has never
been reviewed by a clinician, it was not built for use on real patients or real records, and
nothing here should be used to make a care decision.

Everything in this repository runs offline. There are no runtime dependencies beyond the
Python standard library, and no test or continuous integration step contacts a network.

## What it does

Given a transcript of a visit and a note written from that visit, the tool splits the note into
sentences and decides, for each sentence, whether the conversation supports it. For every
sentence it reports a verdict, the turn numbers it used as evidence, and the reason.

Two checkers are included.

1. **deterministic** matches a sentence to turns by weighted lexical overlap and then applies
   three hard rules. A medication named in the note must appear in the conversation. Every
   quantity must match exactly, so a note that turns 10 mg into 20 mg is unsupported. The
   polarity of a shared term must agree, so "denies chest pain" is supported by "no chest pain"
   and is not supported by "has chest pain". Questions are treated as raising a term rather
   than asserting it, so "any dizziness?" is never read as evidence that dizziness is present.
2. **mock-judge** is an overlap only baseline with none of those three rules. It exists so the
   pluggable judge interface is exercised by the tests and so the deterministic checker has a
   weaker reference point to be measured against. It is not a language model.

A judge that calls a language model is included as an optional extra. It is off by default, it
is not in the registry the evaluation uses, and it raises before opening a socket unless three
environment variables are set. No credential is stored in this repository and the tool never
prints one.

## The data and the answer key

The 30 transcripts in `data/transcripts/`, the 30 notes in `data/notes/`, and the answer key in
`data/labels.json` are entirely synthetic and were **written by hand by me. None of them was
produced by a model.** I wrote them so that this repository can be published without touching
anyone's record, and so that there is a known answer to measure against.

How I authored them:

- I wrote 30 short conversations of 4 to 12 turns each, spread over four common visit types:
  refill, follow up, new symptom, and lab result question. Speakers are only "doctor" and
  "patient". There are no names, no clinic, no dates of birth, no identifiers, and no phone
  numbers.
- For each conversation I then wrote the note a summariser might produce, 4 to 8 sentences
  long, in the clipped style notes are usually written in.
- In 10 of the 30 notes I planted exactly one deliberately unsupported sentence, choosing the
  error from four kinds that matter in practice: an invented dose, an invented symptom, a wrong
  duration, and a wrong medication. The remaining 20 notes are fully supported.
- For every sentence of every note I recorded, by hand, whether the conversation supports it
  and which turn numbers support it. That record is `data/labels.json`.

The key holds 169 sentences, of which 10 are unsupported. The planted errors break down as
three invented symptoms, three wrong durations, two invented doses, and two wrong medications.

## How to run

From a fresh clone:

```bash
pip install -r requirements-dev.txt && pytest
```

Check one note against one conversation:

```bash
python -m grounding check --transcript data/transcripts/t08.json --note data/notes/n08.json
```

```
[2] UNSUPPORTED
    sentence: Patient takes levothyroxine 100 mcg every morning.
    evidence turns: 3
    reason: quantity not supported: 100 mcg
```

The command exits with status 1 when at least one sentence is unsupported, so it can be used as
a gate. Pass `--checker mock-judge` to run the baseline instead. A note can also be a plain
text file rather than JSON.

Score both checkers over all 30 notes against the answer key:

```bash
python -m grounding evaluate --out results/RESULTS.md
```

## Measured results

These are the numbers the evaluation actually produced on the 30 notes in this repository.
The full report, including every individual miss, is in
[results/RESULTS.md](results/RESULTS.md). The positive class is "unsupported", because the job
of the tool is to flag claims the conversation does not support. So a false positive is a
supported sentence that was wrongly flagged, and a false negative is an unsupported sentence
that was missed.

| Checker | Sentences | Precision | Recall | F1 | True positives | False positives | False negatives | True negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deterministic | 169 | 0.909 | 1.000 | 0.952 | 10 | 1 | 0 | 158 |
| mock-judge | 169 | 1.000 | 0.500 | 0.667 | 5 | 0 | 5 | 159 |

Detection by error type:

| Error type | deterministic | mock-judge |
| --- | --- | --- |
| invented dose | 2 of 2 | 1 of 2 |
| invented symptom | 3 of 3 | 2 of 3 |
| wrong duration | 3 of 3 | 1 of 3 |
| wrong medication | 2 of 2 | 1 of 2 |

Evidence turns, measured as the share of correctly supported sentences where the reported
evidence turns overlap the turns recorded in the key: 157 of 158 for the deterministic checker
and 158 of 159 for the baseline.

The gap between the two rows is the measured contribution of the quantity, medication and
negation rules. Without them the baseline misses half of the planted errors, including every
dose and duration change it was shown.

## Limitations

These are real and they bound what the numbers above mean.

- **The data is synthetic and I wrote it.** The conversations are short, clean, and free of the
  interruptions, crosstalk and transcription noise of a real recording. An author who writes
  both the data and the checker will produce data the checker handles well, however carefully
  they try not to.
- **The sample is small.** 30 notes, 169 sentences, and only 10 planted errors. Each planted
  error is worth 0.1 of recall on its own, so these figures carry wide uncertainty and should
  not be read as a general accuracy estimate.
- **The thresholds were chosen while looking at this same data.** There is no held out split,
  so the reported numbers are an upper estimate of how the checker would do on notes it had
  never been tuned against.
- **The lexical checker misses paraphrase.** Matching is over stemmed tokens with no synonym
  list and no embedding, so a note that says "dyspnoea" where the patient said "short of
  breath", or "hypertension" where the conversation said "blood pressure", loses the overlap
  and is flagged even though it is faithful. The measured false positive below is a milder
  version of this.
- **One measured false positive, and it is instructive.** In note n27 the sentence "Lower back
  pain has been present for about ten days." is flagged. The conversation says the back has
  been "hurting", not that there is "pain", while elsewhere the patient says "no pain down the
  leg". The checker compares polarity token by token, so it sees "pain" denied and never
  affirmed, and reports a negation mismatch. The sentence is faithful. Polarity that is scoped
  to a whole token rather than to the body site is the cause, and fixing it properly needs more
  than a word list.
- **Negation is token level.** A term that is denied in one part of a visit and affirmed in
  another is handled by a rule that accepts the sentence if any turn agrees with it, which is
  the right call often enough to be useful and wrong often enough to matter.
- **Wrong medication detection leans on a fixed list of generic names.** A drug outside that
  list is not checked by the medication rule, and a swap to a different drug that the same
  conversation also mentions would not be caught.
- **The mock judge is not a language model.** It is a deliberately weak lexical baseline. The
  comparison in the results table is a comparison against overlap matching, not against a
  model.
- **No clinical validity and no compliance claim.** Nothing here was validated clinically.
  There is no claim of clinical accuracy, no claim of HIPAA compliance, and no suggestion that
  this is fit for production or for any clinical setting. No real patient data is present.

## Repository layout

```
grounding/
  text.py        tokenisation, stemming, stopwords, units, medication names
  sentences.py   sentence splitting that survives decimal doses
  evidence.py    quantity extraction and negation scope
  checker.py     the deterministic checker
  judges.py      the judge interface, the mock baseline, the optional model judge
  datasets.py    loaders and validation for transcripts, notes and the key
  evaluate.py    metrics and the markdown report
  cli.py         the check and evaluate commands
data/
  transcripts/   30 synthetic conversations
  notes/         30 notes written from those conversations
  labels.json    the per sentence answer key
results/
  RESULTS.md     the measured output of the evaluation
tests/           72 tests, no network
```

## Honest label

I built this as a verification study: a way to ask how far a purely deterministic check can go
at catching unsupported claims in a generated note, and to measure the answer on data with a
known key rather than assert it. The result I care about is in the limitations as much as in
the table. The checker caught all 10 planted errors and raised 1 false alarm on 158 supported
sentences, and the one false alarm shows exactly where lexical matching stops being enough.
