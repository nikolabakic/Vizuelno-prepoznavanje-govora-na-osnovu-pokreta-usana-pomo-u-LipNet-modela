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
| [`decoder_results_v1.json`](decoder_results_v1.json) | Phase 7 konfiguracije, metrike, bootstrap intervali i analiza pozicija |
| [`decoder_predictions_v1.json`](decoder_predictions_v1.json) | svih 540 test referenci i predikcija za tri dekodera |

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

## Faza 07 — finalni decoder

Faza 07 je završena u izvršenom notebooku
[`playground/07_faza_7_decoder_search.ipynb`](../../playground/07_faza_7_decoder_search.ipynb).
Greedy kontrola je reprodukovala rezultat faze 05, nakon čega su validation-only
izbor i zaključana test evaluacija dali sledeće rezultate:

| Dekoder | Test WER | Test CER | Exact match |
|---|---:|---:|---:|
| Greedy | 45,25% | 18,24% | 0,00% |
| Prefix beam bez LM-a | 44,88% | 18,10% | 0,00% |
| Prefix beam + train-only 5-gram LM | **41,20%** | **14,70%** | **0,37%** |

Izabrana LM konfiguracija koristi beam width 50, `α = 1,0` i `β = 0,5`.
Paired bootstrap potvrđuje smanjenje WER-a od 4,04 procentna poena, sa 95%
intervalom poverenja od -4,81 do -3,30 procentnih poena. Rezultatni i prediction
artefakt dostupni su direktno u ovom folderu, dok su njihove izvorne kopije i
grafikon `decoder_metrics_v1.png` sačuvani uz checkpoint na Google Drive-u.

## Faza 08 — figure za izveštaj

Konsolidovani notebook 08 koristi gornje sanitizovane artefakte i čuva sedam
izvedenih PNG prikaza u `MyDrive/LipNet/phase8_report/`. Figure se ne tretiraju
kao novi eksperiment: dataset, robustnost i decoder vrednosti dolaze iz Faza
03–07, dok se slot-konfuzije i kvalitativni primeri računaju iz sačuvanih 540
test predikcija. Status završnog Colab izvršavanja prati se u
[`docs/provera-rezultata.md`](../provera-rezultata.md).

## Poreklo i integritet

| Lokalni fajl | Google Drive izvor |
|---|---|
| `phase3_dataset_audit.json` | `MyDrive/LipNet/phase3_dataset_audit.json` |
| `phase4_transfer_ctc_audit.json` | `MyDrive/LipNet/phase4_transfer_ctc_audit.json` |
| `phase5_results.json` | `MyDrive/LipNet/phase5_length_aware_v2/results.json` |
| `phase6_robustness_results.json` | `MyDrive/LipNet/phase5_length_aware_v2/robustness_results.json` |
| `decoder_results_v1.json` | `MyDrive/LipNet/phase5_length_aware_v2/decoder_results_v1.json` |
| `decoder_predictions_v1.json` | `MyDrive/LipNet/phase5_length_aware_v2/decoder_predictions_v1.json` |

Trening i eksperimenti robustnosti koriste isti izabrani checkpoint:

```text
SHA-256: 203c2707b5c327c8b164ab573f5550390def3aacf0ff190fc9bd760745e2f9c8
```

Baseline u eksperimentu robustnosti ponovo je izračunat pre perturbacija i
poklapa se sa rezultatom iz faze 5. Širi trag provere dostupan je u
[`docs/provera-rezultata.md`](../provera-rezultata.md).
