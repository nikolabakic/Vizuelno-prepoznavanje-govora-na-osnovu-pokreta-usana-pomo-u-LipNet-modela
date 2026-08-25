# Testovi

Folder `tests` sadrži brze CPU testove za najvažnije ugovore finalnog sistema.
Za njihovo pokretanje nisu potrebni AI-SPEAK podaci, Google Drive, GPU ni pravi
checkpoint modela.

## Pokretanje

```powershell
uv sync --frozen --all-groups
uv run pytest -q
```

Pojedinačni modul može se pokrenuti navođenjem putanje, na primer:

```powershell
uv run pytest tests/test_decoder.py -q
```

## Pokrivenost

| Fajl | Šta proverava |
|---|---|
| [`test_vipl_phases.py`](test_vipl_phases.py) | speaker split, srpski vokabular, alignment parser, variable-length batch i transfer težina |
| [`test_phase5.py`](test_phase5.py) | CTC trening, corpus metrike, checkpoint resume i usklađenost notebookova sa generatorom |
| [`test_decoder.py`](test_decoder.py) | egzaktno CTC prefix beam dekodiranje i karakterni jezički model |
| [`test_evaluation.py`](test_evaluation.py) | paired bootstrap i strukturisana analiza grešaka |

Testovi namerno koriste male sintetičke ulaze. GPU validacija kompletnog modela
i numerički rezultati eksperimenata dokumentovani su u
[`docs/results`](../docs/results/README.md).
