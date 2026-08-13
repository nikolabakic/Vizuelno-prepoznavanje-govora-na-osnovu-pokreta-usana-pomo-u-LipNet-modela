# Faza 2: manifest, QA, split i mouth ROI

Ova faza koristi isključivo lokalni korpus u `processed/processed/`. Generisani artefakti ostaju u ignorisanom direktorijumu `artifacts/phase2/`, jer sadrže lokalne putanje i prepoznatljive mouth crop-ove.

## 1. Instalacija i manifest

```powershell
uv sync
uv run python -m app.phase2 build
```

Ako Windows/OneDrive politika zabrani pokretanje `.venv/Scripts/python.exe`, projekat treba klonirati u običan lokalni direktorijum (na primer `C:\dev\lipnet-serbian`) ili podesiti `UV_PROJECT_ENVIRONMENT` na lokaciju na kojoj je izvršavanje dozvoljeno.

Komanda uparuje svaki MP4 sa ALIGN fajlom, proverava identitete, čita video metapodatke i transkript, pa zapisuje:

- `manifest.csv` — putanje, govornik, transkript, trajanje, frejmovi i CTC dužine;
- `vocab.json` — 29 klasa: blank, razmak i 27 srpskih latiničnih karaktera;
- `split.json` — deterministički speaker-disjunktni train/validation/test split (`seed=42`).

Relativne putanje u manifestu su prenosive u odnosu na koren korpusa. Validacija odbija neuparene fajlove, nepoznate karaktere, prazne transkripte i targete koji su duži od dostupne vremenske ose. Vremena govornih tokena se strogo proveravaju; neispravna granica pomoćnog `sil` tokena ne odbacuje CTC target (takva anomalija postoji u `spk01_019.align`).

## 2. Stabilan mouth ROI

```powershell
uv run python -m app.phase2 roi
```

MediaPipe Face Mesh uzorkuje pet vremenski raspoređenih frejmova. Za svaki klip koristi medijanu detektovanih lip bounding box-ova, proširenu i prilagođenu odnosu `2:1`, što odgovara kasnijem crop-u `128×64`. Ako ceo klip nema detekciju, koristi se medijana uspešnih ROI-jeva istog govornika; tek zatim globalna medijana.

Rezultati su:

- `roi.csv` — normalizovane koordinate, broj uspešnih detekcija i izvor ROI-ja;
- `roi_qa.jpg` — kontakt-list sa svim govornicima i nekoliko najdužih klipova (najmanje 30);
- `roi_qa.json` — lista prikazanih primera i polje `manual_review_complete`.

Obavezno ručno otvoriti `roi_qa.jpg`. Proveriti da usne nisu odsečene, da crop ne skače van lica i posebno pregledati redove označene sa `speaker_median` ili `global_median`. Tek nakon pregleda promeniti `manual_review_complete` u `true`.

## 3. Testovi

```powershell
uv run python -m unittest discover -s tests -v
```

Testovi pokrivaju ALIGN parsiranje, 29 CTC klasa, speaker-disjunktnost i odbijanje karaktera van vokabulara. Komanda `build` dodatno radi integracionu proveru nad svih 3.957 lokalnih primera.
