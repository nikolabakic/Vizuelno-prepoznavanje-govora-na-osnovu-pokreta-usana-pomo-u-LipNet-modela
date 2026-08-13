# VIPL upstream evidencija i minimalna odstupanja

## Pinovani izvor

| Polje | Vrednost |
|---|---|
| Repository | `https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch` |
| Branch | `master` |
| Commit | `40209e09c49553c00c25c7d41faa3706aea3c625` |
| Commit datum | `2022-09-21T14:35:02+08:00` |
| Licenca | MIT deklaracija u upstream README; sačuvana kao `lipnet/LICENSE.vipl` |

Pin je read-only proverljiv komandom:

```bash
git ls-remote https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch.git refs/heads/master
```

## Inventar

| Upstream | Lokalno odredište | Uloga |
|---|---|---|
| `model.py` | `lipnet/model.py` | 3D CNN, dva BiGRU sloja i CTC head |
| `dataset.py` | `lipnet/dataset.py` | GRID učitavanje/dekoder/metrike + mali srpski adapter |
| `cvtransforms.py` | `lipnet/cvtransforms.py` | horizontalni flip i `/255.0` normalizacija |
| `main.py` | `lipnet/train.py` | checkpoint load, greedy decode, CTC loss i smoke backward |
| `demo.py` | `lipnet/demo.py` | FFmpeg 25 fps, face alignment, afina transformacija, mouth crop |
| `options.py` | `lipnet/options.py` | zajednički baseline parametri i pinovani checkpoint URL |
| `data/*.txt` | kloniran upstream u Fazi 1 | originalni GRID indeksi ostaju upstream artefakt |
| `pretrain/*.pt` | Colab privremeni/Drive checkpoint | veliki fajl se ne šalje u Git |

## Odstupanja po fazama

| Faza | Upstream | Lokalno | Razlog | Parity/provera |
|---:|---|---|---|---|
| 0 | grana `master` | tačan SHA + licenca + inventar | reproducibilno poreklo | notebook proverava remote SHA i potrebne fajlove |
| 1 | `LipNet(dropout_p)` sa 28 klasa | `LipNet(dropout_p, num_classes=28)` | isti model mora kasnije prihvatiti srpski head | upstream i lokalni model učitavaju isti checkpoint i moraju dati isti logits/dekodirani tekst |
| 1 | direktni `.cuda()` i stariji import obrasci | eksplicitni `torch.device` i `map_location` | aktuelni Colab/PyTorch | strict checkpoint audit; identičan decode na istom GRID tensoru |
| 1 | shell string za FFmpeg | `subprocess.run([...], check=True)` | bezbedne putanje i vidljive greške | oblik/opseg ulaza `(C,T,64,128)`, `[0,1]` |
| 2 | `LandmarksType._2D` | `LandmarksType.TWO_D` uz fallback | novi `face-alignment` API | isti kanonski landmark-i, afina transformacija i `160x80 -> 128x64` crop |
| 2 | podrazumevani SFD detector u jednokratnom demo-u | BlazeFace detector za korpus | SFD bi za 3.957 lokalnih klipova zahtevao približno 29 sati; BlazeFace smoke obrada je praktična za Colab | isti 2D FAN landmark model i VIPL geometrija; QA normalnih/graničnih klipova svakog govornika |
| 2 | demo vraća tensor jednog MP4-a | numerisani JPEG folderi `spkXX/video/video_a/sample/` | upstream Dataset trening čita JPEG sekvence | `MyDataset._load_vid` čita rezultat i vraća `(T,64,128,3)` |
| 2 | propušten landmark znači preskočen frejm | isto, uz brojanje i `failed_clips.log`/QA | neuspeh mora biti vidljiv bez trening manifesta | QA sheet i runtime log; Dataset ne čita log |
| 3 | space-separated GRID `.align`, velika slova | tab-separated AI-SPEAK `.align`, NFC mala slova | format lokalnog skupa i srpski alfabet | parser test i `txt2arr -> arr2txt` round-trip |
| 3 | fiksni `vid_pad=75`, `txt_pad=200` | batch-local dinamički padding | promenljive dužine klipova | dva različita klipa u istom batch-u, realni `vid_len`/`txt_len` |
| 3 | tekstualna lista svakog GRID uzorka | verzionisane speaker liste + runtime discovery | speaker-disjoint split bez manifesta | disjunktnost se validira pri importu `data/splits.py` |
| 4 | fiksni `FC: 512 -> 28` | parametrizovan `FC`, srpski `512 -> 29` | blank + razmak + 27 znakova iz lokalnog korpusa | shape-filtered audit mora preskočiti samo `FC.weight`/`FC.bias` |
| 4 | CTC u petlji sa `.cuda()` | eksplicitni device i provera minimalnih CTC koraka | savremeni API i rani shape error | jedan mali batch radi forward, konačan loss i backward bez NaN gradijenata |

## Legacy izolacija

`app/data.py`, `app/preprocessing.py`, `app/phase2.py`, `app/gpu_roi.py`, stari
Faza 1/2 notebookovi i dokumenti ostaju samo radi istorije rada. Aktivni kod u
`lipnet/`, `scripts/prepare_ai_speak.py` i novim `00_...`–`04_...` notebookovima
ne uvozi `app`, ne čita `manifest.csv`, `roi.csv`, `roi_qa.json`, `vocab.json`
niti `split.json` i ne koristi statičan/median ROI.
