# Finalni izveštaj

Folder `report` sadrži završni rad projekta i verzionisane slike korišćene u
njemu. Izveštaj objedinjuje metod, eksperimentalni protokol, rezultate Faza
03–08, ograničenja i zaključak.

## Formati

| Fajl | Namena |
|---|---|
| [`finalni-izvestaj-lipnet.pdf`](finalni-izvestaj-lipnet.pdf) | finalna verzija spremna za čitanje i predaju |
| [`finalni-izvestaj-lipnet.html`](finalni-izvestaj-lipnet.html) | izvorna, pregledljiva HTML verzija istog izveštaja |

## Slike

| Fajl | Sadržaj |
|---|---|
| [`01_dataset_split.png`](assets/01_dataset_split.png) | speaker-disjoint podela skupa |
| [`02_gpu_demo_mouth_frames.png`](assets/02_gpu_demo_mouth_frames.png) | mouth frejmovi GPU demonstracionog primera 42 |
| [`03_robustness.png`](assets/03_robustness.png) | uticaj rezolucije, zamućenja i pomeranja crop-a |
| [`04_decoder_comparison.png`](assets/04_decoder_comparison.png) | poređenje greedy, beam i beam+5-gram dekodera |
| [`05_slot_accuracy.png`](assets/05_slot_accuracy.png) | tačnost po šest AI-SPEAK pozicija |
| [`06_slot_confusion_matrices.png`](assets/06_slot_confusion_matrices.png) | normalizovane matrice konfuzije finalnog dekodera |
| [`07_letter_substitutions.png`](assets/07_letter_substitutions.png) | najčešće zamene izolovanih slova |
| [`08_training_history.png`](assets/08_training_history.png) | istorija treninga i izbor najboljeg checkpoint-a |

Prvih sedam slika potiče iz izvršenog
[`08_faza_8_konsolidovani_notebook.ipynb`](../playground/08_faza_8_konsolidovani_notebook.ipynb),
dok je istorija treninga izvedena iz potvrđenog rezultata Faze 05. Privatni
snimci, mouth frejmovi i checkpoint modela nisu deo ovog foldera.

Brojčani izvor istine ostaju sanitizovani JSON artefakti u
[`docs/results`](../docs/results/README.md), a trag validacije nalazi se u
[`docs/provera-rezultata.md`](../docs/provera-rezultata.md).
