# Faza 2: GPU ROI u Google Colab-u

Najjednostavniji način je da uploaduješ i pokreneš gotov [samostalni Colab notebook](../playground/lipnet_faza2_gpu_colab.ipynb). On ne zahteva da repo prethodno bude na GitHub-u: dovoljan je `processed.zip`. Uputstvo ispod ostaje kao ručna alternativa.

Manifest, split i vokabular su već generisani i validirani lokalno za svih 3.957 primera. Na GPU prebacujemo samo landmark/ROI korak. Python MediaPipe pipeline nije CUDA rešenje; ovaj tok koristi `face-alignment` sa eksplicitnim `device="cuda"`.

## Potrebno

- Colab runtime sa T4 GPU-om;
- ovaj repozitorijum na GitHub-u ili Google Drive-u;
- `processed.zip` na Google Drive-u (oko 3,8 GiB podataka nakon raspakivanja);
- dovoljno Colab diska za ZIP i raspakovanu kopiju (preporučeno najmanje 10 GiB slobodno).

Nemoj obrađivati MP4 fajlove direktno sa montiranog Drive-a: kopiranje ZIP-a jednom i raspakivanje u `/content` je mnogo brže od hiljada malih Drive čitanja.

## Colab ćelije

Prvo izaberi **Runtime → Change runtime type → T4 GPU**, pa pokreni ćelije redom.

### 1. Provera GPU-a i Drive

```python
import torch
assert torch.cuda.is_available(), "Uključi T4 GPU u Runtime postavkama"
print(torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount("/content/drive")
```

### 2. Repo i lokalna kopija korpusa

Ako je repo na GitHub-u:

```python
!git clone URL_TVOG_REPOZITORIJUMA /content/lipnet-serbian
```

Ako je repo već na Drive-u, kopiraj ga u `/content` ili postavi `REPO` na njegovu putanju. Zatim prilagodi samo `ZIP_ON_DRIVE`:

```python
from pathlib import Path
import shutil, zipfile

REPO = Path("/content/lipnet-serbian")
ZIP_ON_DRIVE = Path("/content/drive/MyDrive/processed.zip")
LOCAL_ZIP = Path("/content/processed.zip")
DATA_DIR = Path("/content/data")

assert REPO.exists(), REPO
assert ZIP_ON_DRIVE.exists(), ZIP_ON_DRIVE
shutil.copy2(ZIP_ON_DRIVE, LOCAL_ZIP)
DATA_DIR.mkdir(exist_ok=True)
with zipfile.ZipFile(LOCAL_ZIP) as archive:
    archive.extractall(DATA_DIR)

candidates = [DATA_DIR / "processed" / "processed", DATA_DIR / "processed"]
CORPUS = next((path for path in candidates if list(path.glob("spk*/alignment/*.align"))), None)
assert CORPUS is not None, "Nije pronađen spk*/alignment/*.align u arhivi"
print(CORPUS)
```

### 3. Instalacija GPU landmark biblioteke

Colab već sadrži CUDA PyTorch; ne instalira se CUDA ručno.

```python
%cd /content/lipnet-serbian
!python -m pip install -q face-alignment==1.5.0
```

### 4. Manifest i GPU ROI

`build` je kratak CPU korak koji čita video metapodatke. Landmark obrada u drugoj komandi koristi CUDA.

```python
!python -m app.phase2 build --corpus "{CORPUS}" --output /content/phase2
!python -m app.gpu_roi --corpus "{CORPUS}" --input /content/phase2 --samples 5 --max-side 640 --face-detector sfd
```

SFD je sporiji, ali najprecizniji podržani detektor. Ako je i dalje presporo, ponovi samo drugu komandu sa `--face-detector blazeface`; kvalitet obavezno proceni na QA slici.

### 5. Pregled i čuvanje rezultata

```python
from IPython.display import display, Image
display(Image(filename="/content/phase2/roi_qa.jpg"))
```

Ako su svih 30 crop-ova dobri, u `roi_qa.json` promeni `manual_review_complete` na `true`, pa sačuvaj sve na Drive:

```python
import json, shutil
qa_path = Path("/content/phase2/roi_qa.json")
qa = json.loads(qa_path.read_text(encoding="utf-8"))
qa["manual_review_complete"] = True
qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

archive = shutil.make_archive("/content/faza2_artifacts", "zip", "/content/phase2")
shutil.copy2(archive, "/content/drive/MyDrive/faza2_artifacts.zip")
print("Sačuvano: /content/drive/MyDrive/faza2_artifacts.zip")
```

Očekivani ZIP sadrži `manifest.csv`, `split.json`, `vocab.json`, `roi.csv`, `roi_qa.jpg` i `roi_qa.json`. Nemoj ga slati u Git ako QA slika otkriva identitet; čuvaj ga uz lokalne trening artefakte.

`face-alignment` zvanično podržava CUDA i izbor SFD/BlazeFace/RetinaFace detektora; SFD je naveden kao najprecizniji, dok su drugi brži. Izvor: [zvanični face-alignment repozitorijum](https://github.com/1adrianb/face-alignment).
