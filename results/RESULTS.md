# Measured results

Produced by `python -m grounding evaluate`. The positive class is a sentence the conversation does not support, so a false positive is a supported sentence that was flagged and a false negative is an unsupported sentence that was missed.

## Dataset

| Quantity | Count |
| --- | --- |
| notes | 30 |
| sentences | 169 |
| unsupported sentences | 10 |
| notes with at least one unsupported sentence | 10 |

## Checker results

| Checker | Sentences | Precision | Recall | F1 | True positives | False positives | False negatives | True negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deterministic | 169 | 0.909 | 1.000 | 0.952 | 10 | 1 | 0 | 158 |
| mock-judge | 169 | 1.000 | 0.500 | 0.667 | 5 | 0 | 5 | 159 |

## Evidence turns

Share of correctly supported sentences where the reported evidence turns overlap the turns recorded in the answer key.

| Checker | Sentences compared | Overlapping | Rate |
| --- | --- | --- | --- |
| deterministic | 158 | 157 | 0.994 |
| mock-judge | 159 | 158 | 0.994 |

## Detection by error type

| Error type | deterministic | mock-judge |
| --- | --- | --- |
| invented dose | 2 of 2 | 1 of 2 |
| invented symptom | 3 of 3 | 2 of 3 |
| wrong duration | 3 of 3 | 1 of 3 |
| wrong medication | 2 of 2 | 1 of 2 |

## Individual misses

### deterministic

| Kind | Note | Sentence | Text | Detail |
| --- | --- | --- | --- | --- |
| false positive | n27 | 0 | Lower back pain has been present for about ten days. | negation mismatch on: pain |

### mock-judge

| Kind | Note | Sentence | Text | Detail |
| --- | --- | --- | --- | --- |
| false negative | n03 | 4 | Patient also reports vomiting for two days. | invented symptom |
| false negative | n08 | 2 | Patient takes levothyroxine 100 mcg every morning. | invented dose |
| false negative | n14 | 5 | Patient denies stomach upset on atorvastatin 500 mg twice daily. | wrong medication |
| false negative | n21 | 5 | Warfarin was refilled for six months. | wrong duration |
| false negative | n29 | 2 | Amoxicillin 500 mg three times daily for 14 days was sent. | wrong duration |
