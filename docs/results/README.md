# Potvrđeni artefakti pre finalnog izveštaja

Ovaj folder sadrži male, sanitizovane JSON kopije rezultata sa Google Drive-a.
Privatni video, mouth frejmovi i checkpoint-i nisu kopirani u Git.

| Fajl | Poreklo | Status |
|---|---|---|
| `phase3_dataset_audit.json` | `MyDrive/LipNet/phase3_dataset_audit.json` | potvrđeno |
| `phase4_transfer_ctc_audit.json` | `MyDrive/LipNet/phase4_transfer_ctc_audit.json` | potvrđeno |
| `phase5_results.json` | `MyDrive/LipNet/phase5_length_aware_v2/results.json` | potvrđeno |
| `phase6_robustness_results.json` | `MyDrive/LipNet/phase5_length_aware_v2/robustness_results.json` | potvrđeno |

Phase 5 i Phase 6 fajlovi navode isti checkpoint SHA-256:

```text
203c2707b5c327c8b164ab573f5550390def3aacf0ff190fc9bd760745e2f9c8
```

Faza 7 će na Drive-u napraviti `decoder_results_v1.json` i
`decoder_predictions_v1.json`. Nakon uspešnog greedy baseline `assert`-a i
validation-only izbora decoder parametara, sanitizovana kopija rezultata treba
da se doda ovde. To je poslednji numerički korak pre konsolidovanog notebooka i
kasnijeg finalnog izveštaja.
