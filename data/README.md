# Podela podataka

Folder `data` sadrži male, verzionisane definicije potrebne za reprodukciju
eksperimenta. Sami AI-SPEAK snimci se ne čuvaju u repozitorijumu.

## Sadržaj

| Fajl | Namena |
|---|---|
| [`splits.py`](splits.py) | definiše train, validation i test govornike |
| [`__init__.py`](__init__.py) | označava folder kao Python paket |

## Speaker-disjoint podela

Podela je napravljena po govornicima, tako da se ista osoba nikada ne pojavljuje
u više skupova.

| Skup | Govornici | Primeri |
|---|---:|---:|
| Train | 16 | 2.877 |
| Validation | 3 | 540 |
| Test | 3 | 540 |

`validate_splits()` se izvršava pri uvozu modula i prijavljuje grešku ako
postoji presek između skupova. Dataset zatim dinamički pronalazi primere za
govornike navedene u `SPLITS`; poseban manifest po uzorku nije potreban.

```python
from data.splits import SPLITS

train_speakers = SPLITS["train"]
```

Detaljan audit broja primera dostupan je u
[`docs/results/phase3_dataset_audit.json`](../docs/results/phase3_dataset_audit.json).
