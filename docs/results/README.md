# Rezultati eksperimenta

Ovaj folder sadrži male, sanitizovane JSON artefakte koji predstavljaju
proverene ulaze i rezultate finalnog eksperimentalnog pipeline-a. Privatni
video-snimci, mouth frejmovi i checkpoint-i ostaju van Git repozitorijuma.

## Sadržaj

| Fajl | Sadržaj |
|---|---|
| [`phase3_dataset_audit.json`](phase3_dataset_audit.json) | veličine splitova, broj klasa i provera variable-length batch-a |
| [`phase4_transfer_ctc_audit.json`](phase4_transfer_ctc_audit.json) | transfer VIPL težina, oblik izlaza i CTC backward provera |
| [`phase5_results.json`](phase5_results.json) | konfiguracija treninga, validation/test metrike i primeri predikcija |
| [`phase6_robustness_results.json`](phase6_robustness_results.json) | rezultati za rezoluciju, blur i pomeranje regiona usana |

## Glavne metrike

| Eksperiment | WER | CER |
|---|---:|---:|
| Validation baseline | 49,72% | 22,02% |
| Test baseline | **45,25%** | **18,24%** |
| 96 × 48 | 45,77% | 18,59% |
| 64 × 32 | 45,90% | 18,70% |
| Gaussian blur | 45,96% | 18,73% |
| Pomeranje crop-a | 45,34% | 18,34% |

WER i CER su corpus-level metrike: edit greške se sabiraju nad celim skupom, a
zatim dele ukupnim brojem referentnih reči, odnosno karaktera.

## Poreklo i integritet

| Lokalni fajl | Google Drive izvor |
|---|---|
| `phase3_dataset_audit.json` | `MyDrive/LipNet/phase3_dataset_audit.json` |
| `phase4_transfer_ctc_audit.json` | `MyDrive/LipNet/phase4_transfer_ctc_audit.json` |
| `phase5_results.json` | `MyDrive/LipNet/phase5_length_aware_v2/results.json` |
| `phase6_robustness_results.json` | `MyDrive/LipNet/phase5_length_aware_v2/robustness_results.json` |

Trening i eksperimenti robustnosti koriste isti izabrani checkpoint:

```text
SHA-256: 203c2707b5c327c8b164ab573f5550390def3aacf0ff190fc9bd760745e2f9c8
```

Baseline u eksperimentu robustnosti ponovo je izračunat pre perturbacija i
poklapa se sa rezultatom iz faze 5. Širi trag provere dostupan je u
[`docs/provera-rezultata.md`](../provera-rezultata.md).
