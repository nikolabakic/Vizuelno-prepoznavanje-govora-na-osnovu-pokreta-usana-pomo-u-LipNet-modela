# LipNet implementacija

Folder `lipnet` sadrži jezgro finalnog rešenja: model, pripremu ulaza, Dataset
adapter, fine-tuning, CTC dekodiranje i evaluaciju. Implementacija polazi od
VIPL LipNet-a, a zatim uvodi podršku za srpski AI-SPEAK korpus i promenljive
dužine video-sekvenci.

## Arhitektura

```text
video (B, C, T, H, W)
        │
        ▼
3 × Conv3D + MaxPool3D
        │
        ▼
2 × BiGRU sa realnim dužinama sekvenci
        │
        ▼
linearni sloj sa 29 CTC klasa
        │
        ▼
greedy ili prefix beam dekodiranje
```

## Sadržaj

| Fajl | Odgovornost |
|---|---|
| [`model.py`](model.py) | 3D CNN + BiGRU + CTC LipNet arhitektura |
| [`dataset.py`](dataset.py) | GRID kompatibilnost, srpski vokabular, alignment parser i variable-length batch |
| [`demo.py`](demo.py) | dekodiranje videa, poravnanje lica i izdvajanje regiona usana |
| [`train.py`](train.py) | transfer težina, CTC loss, trening, metrike i checkpoint-i |
| [`decoder.py`](decoder.py) | CTC prefix beam search i karakterni n-gram jezički model |
| [`evaluation.py`](evaluation.py) | paired bootstrap, analiza grešaka po pozicijama i poravnate slot-konfuzije |
| [`cvtransforms.py`](cvtransforms.py) | video transformacije i normalizacija |
| [`options.py`](options.py) | zajednički baseline parametri i pinovani VIPL checkpoint |
| [`LICENSE.vipl`](LICENSE.vipl) | licenca izvornog VIPL koda |

## Ključna prilagođavanja

- izlazni sloj koristi 29 klasa: CTC blank, razmak i karaktere srpske latinice;
- batch-evi podržavaju klipove različitih dužina;
- padded vreme se maskira posle konvolucionih blokova;
- oba BiGRU sloja koriste realne dužine pomoću packed sekvenci;
- WER i CER se računaju na nivou celog korpusa;
- VIPL checkpoint prenosi sve kompatibilne težine, dok se CTC head inicijalizuje
  za novi vokabular.

Tačna odstupanja od pinovane VIPL verzije dokumentovana su u
[`docs/upstream-diff.md`](../docs/upstream-diff.md), a ponašanje modula je
pokriveno testovima u folderu [`tests`](../tests/README.md).
