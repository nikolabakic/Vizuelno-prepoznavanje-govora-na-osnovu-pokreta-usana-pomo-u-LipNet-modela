# VIPL-first roadmap za LipNet projekat

Datum revizije: 18. avgust 2026.

## Status implementacije

Kod i Colab notebookovi za Faze 0–5 napisani su do 18. avgusta 2026. Prihvatni
kriterijumi koji zahtevaju GRID/AI-SPEAK podatke ili GPU namerno još nisu
označeni kao izvršeni; potvrđuju se redom u `playground/00_...` do `05_...`.

| Faza | Kod/notebook | Izvršni status |
|---:|---|---|
| 0 | pin, licenca, inventar i legacy isolation check | spremno za CPU Run all |
| 1 | originalni VIPL naspram lokalnog GRID parity toka | spremno za Colab GPU |
| 2 | VIPL demo MP4 → mouth JPEG + QA/log | spremno za Colab GPU |
| 3 | srpski Dataset, split i promenljivi collate | spremno posle artefakta Faze 2 |
| 4 | 29-klasni head, transfer audit i CTC backward | spremno posle Faze 3, Colab GPU |
| 5 | postepeni fine-tuning, resume, best-WER izbor i test | spremno posle Faze 4, Colab GPU |

## 1. Nova odluka

Projekat se ponovo radi polazeći od referentnog repozitorijuma
[VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch).
Upstream kod je izvor istine za arhitekturu, format ulaza, transformacije, trening,
CTC dekodiranje, metrike, učitavanje checkpoint-a i preprocessing. Lokalni kod se
piše samo tamo gde VIPL rešenje ne može neposredno da radi sa dostavljenim
AI-SPEAK podskupom ili savremenim Colab/PyTorch okruženjem.

Prethodno implementirani pipeline nije osnova za nastavak rada. Posebno se više
ne koriste:

- `manifest.csv`, `roi.csv`, `roi_qa.json`, `vocab.json` i `split.json` kao ulaz u trening;
- MediaPipe/median-ROI pristup iz `app/preprocessing.py`;
- zasebne komande `app.phase2 build`, `app.phase2 roi` i `app.gpu_roi`;
- postojeći notebook-ovi Faze 1 i Faze 2 kao osnova novog notebook-a;
- prethodna podela na faze u kojoj se prvo gradi sopstvena infrastruktura pa tek
  kasnije dodaje LipNet.

Postojeći fajlovi i artefakti se do početka implementacije smatraju **legacy**
materijalom. Ne prenose se njihovi generisani podaci, pretpostavke niti API-ji u
novi tok. Brisanje ili zamena koda radiće se tek u implementacionoj Fazi 0, nakon
što je referentna verzija VIPL repoa sačuvana i proverena.

## 2. Obavezno upstream-first pravilo

Pre svake implementacione stavke:

1. pronaći odgovarajući fajl i funkciju u VIPL repozitorijumu;
2. prvo preuzeti kod bez izmene ili sa najmanjom mogućom izmenom;
3. dokumentovati svako odstupanje u kratkoj tabeli `upstream -> lokalno -> razlog`;
4. proveriti lokalno ponašanje prema upstream-u na istom malom primeru;
5. ne uvoditi dodatnu apstrakciju, format ili artefakt ako nije potreban za zadatak.

Referentni upstream fajlovi su:

- [`model.py`](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch/blob/master/model.py)
  za 3D CNN + BiGRU + linearni CTC head;
- [`dataset.py`](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch/blob/master/dataset.py)
  za učitavanje frejmova, `.align` anotacija, padding i mapiranje teksta;
- [`cvtransforms.py`](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch/blob/master/cvtransforms.py)
  za horizontalni flip i normalizaciju;
- [`main.py`](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch/blob/master/main.py)
  za DataLoader, CTC loss, greedy decode, WER/CER, trening, validaciju i checkpoint-e;
- [`demo.py`](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch/blob/master/demo.py)
  za MP4 dekodiranje, face alignment, afinu transformaciju i mouth crop `128x64`;
- [`options.py`](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch/blob/master/options.py)
  za konfiguraciju eksperimenta;
- upstream `data/` indeksi i `pretrain/` checkpoint-i za reprodukciju GRID rezultata.

Pre početka izmena mora se zabeležiti tačan upstream commit SHA. Kopirani MIT
licencirani kod mora zadržati originalno obaveštenje o licenci i jasno naveden izvor.

## 3. Šta se preuzima, a šta sme da se menja

| Oblast | VIPL osnova | Lokalna odluka |
|---|---|---|
| Model | `LipNet` iz `model.py` | Preuzeti slojeve, dimenzije i `forward`; promeniti samo broj izlaza za srpski alfabet |
| Inicijalizacija | `_init()` iz `model.py` | Zadržati; novi izlazni sloj inicijalizovati istim pravilom |
| Checkpoint | shape-filtered load iz `main.py` | Zadržati konvolucione i GRU težine; namerno preskočiti nekompatibilni `FC` |
| Video preprocessing | `load_video()` iz `demo.py` | Zadržati face alignment, afinu transformaciju i mouth crop; prilagoditi samo novi API biblioteke i bezbedan poziv FFmpeg/OpenCV-a |
| Ulazna slika | JPEG sekvence, resize `128x64` | AI-SPEAK MP4 jednom pretvoriti u isti upstream folder frejmova; trening ne čita MP4 direktno |
| Normalizacija | deljenje sa `255.0` | Preuzeti bez promene |
| Augmentacija | `HorizontalFlip` | Preuzeti kao baseline; blur i crop jitter dodati tek u eksperimentalnoj fazi |
| Anotacije | uklanjanje `SIL`/`SP`, spajanje reči | Preuzeti tok; promeniti parser samo koliko zahteva tab-separated lokalni `.align` format i mala slova |
| Vokabular | `MyDataset.letters` u kodu | Definisati srpska slova u istom modulu; bez zasebnog `vocab.json` |
| Split | upstream tekstualne liste | Za lokalni korpus koristiti eksplicitne speaker liste u konfiguraciji i dinamičko pronalaženje fajlova; bez bogatog CSV manifesta |
| Batch/padding | `vid_padding=75`, `txt_padding=200` | Prvo reprodukovati bez izmene na GRID-u; za promenljive AI-SPEAK klipove napraviti najmanju potrebnu izmenu collate/padding logike |
| CTC | `nn.CTCLoss()` i stvarne dužine | Preuzeti tok; dodati samo validaciju dužina i savremeni oblik tenzora gde je nužno |
| Dekodiranje | greedy argmax + collapse repeats/blanks | Preuzeti; bez beam search-a, jezičkog modela ili gramatičkog korektora |
| Metrike | upstream WER/CER | Preuzeti i dodati sentence exact match za zahtev izveštaja |
| Trening | Adam, validacija i checkpoint iz `main.py` | Zadržati kao početni baseline; menjati samo device, konfiguraciju i parametre potrebne za Colab |

## 4. Dozvoljena odstupanja

Odstupanje od VIPL koda je dozvoljeno samo kada je pokriveno jednim od sledećih
razloga:

1. **Srpski alfabet:** engleski izlazni sloj ima 28 klasa, dok lokalni karakter-level
   CTC zahteva blank, razmak i srpska latinična slova prisutna u korpusu.
2. **Format lokalnog skupa:** AI-SPEAK podskup je organizovan kao MP4 + tab-separated
   `.align`, dok VIPL trening očekuje foldere JPEG frejmova i GRID anotacije.
3. **Promenljiva dužina:** GRID klipovi imaju 75 frejmova, a lokalni klipovi su duži
   i promenljivi. Batch mora očuvati stvarnu `vid_len` vrednost i dati CTC-u tačne
   dužine.
4. **Savremeno okruženje:** direktni `.cuda()`, stari `face_alignment` enum i stari
   PyTorch obrasci menjaju se samo radi rada na aktuelnom Colab-u i CPU/GPU izboru.

Sve drugo se najpre pokušava sa upstream implementacijom. „Lepši” lokalni dizajn
nije dovoljan razlog za odstupanje.

## 5. Ciljna struktura repozitorijuma

Struktura treba da ostane prepoznatljivo bliska upstream-u:

```text
lipnet/
├── model.py              # VIPL model; samo parametrizovan broj klasa
├── dataset.py            # VIPL Dataset + minimalni srpski adapter
├── cvtransforms.py       # VIPL transformacije + eksperimentalne transformacije
├── train.py              # adaptiran VIPL main.py
├── demo.py               # adaptiran VIPL demo.py
├── options.py            # zajednički baseline parametri
└── LICENSE.vipl          # upstream MIT licenca i poreklo koda
scripts/
├── prepare_grid.py       # samo ako je potrebno za reprodukciju
└── prepare_ai_speak.py   # MP4 -> VIPL-kompatibilni mouth JPEG folderi
data/
└── splits.py             # eksplicitni speaker ID-jevi; bez manifesta sa uzorcima
playground/
└── lipnet_colab.ipynb    # jedan notebook za reprodukciju, trening i demonstraciju
docs/
├── upstream-diff.md      # obavezna evidencija minimalnih odstupanja
└── analiza-i-roadmap.md
```

Nazivi mogu malo da se promene tokom implementacije, ali VIPL odgovornost fajlova
ne treba deliti na novu lokalnu infrastrukturu bez stvarne potrebe.

## 6. Roadmap

### Faza 0 — Čist restart i pinovanje upstream-a

**Cilj:** obezbediti proverljivu osnovu pre novog koda.

- zabeležiti URL, branch i tačan commit SHA VIPL repozitorijuma;
- sačuvati upstream licencu;
- napraviti inventar upstream fajlova i njihovih lokalnih odredišta;
- označiti postojeće `app/data.py`, `app/preprocessing.py`, `app/phase2.py`,
  `app/gpu_roi.py`, Faza 1/2 notebook-ove i generisane Phase 2 artefakte kao legacy;
- ukloniti ih iz aktivnog toka pre nego što se uvede novi kod;
- otvoriti `docs/upstream-diff.md` sa praznom tabelom odstupanja.

**Prihvatni kriterijum:** nijedna naredna faza ne uvozi legacy module niti čita
prethodne manifeste/ROI artefakte.

### Faza 1 — Reprodukcija originalnog VIPL LipNet-a

**Cilj:** dokazati da upstream radi pre bilo kakve srpske adaptacije.

- preuzeti VIPL `model.py`, `dataset.py`, `cvtransforms.py`, `main.py`, `demo.py`
  i `options.py` u radnu strukturu;
- koristiti upstream unseen-speaker checkpoint;
- koristiti upstream GRID mouth frejmove i `.align` format;
- pokrenuti inference i greedy CTC decode na najmanje jednom GRID primeru;
- potvrditi da su sve checkpoint težine učitane;
- izračunati predikciju, CER i WER upstream funkcijama;
- zatim napraviti samo nužne izmene za aktuelni PyTorch/Colab i ponoviti isti test.

**Prihvatni kriterijum:** originalna i minimalno modernizovana verzija daju isti
dekodirani tekst na izabranom primeru. Bez ove tačke se ne prelazi na AI-SPEAK.

### Faza 2 — AI-SPEAK preprocessing po VIPL demo pipeline-u

**Cilj:** pretvoriti lokalne MP4 snimke u isti oblik koji očekuje upstream Dataset.

- dinamički pronaći `spk*/ser/video_a/*.mp4` i odgovarajuće `.align` fajlove;
- za svaki video primeniti tok iz VIPL `demo.py`: dekodiranje na 25 fps, 68
  landmark-a, poravnanje lica, afina transformacija, mouth crop i resize `128x64`;
- čuvati numerisane JPEG frejmove u folder strukturi kompatibilnoj sa
  `MyDataset._load_vid`;
- ne računati jedan statičan ROI po klipu i ne generisati `roi.csv`;
- vizuelno proveriti nekoliko frejmova svakog govornika, uključujući neuspele i
  granične slučajeve;
- zabeležiti neuspele klipove u običan runtime log, ne u novi trening manifest.

**Prihvatni kriterijum:** jedan lokalni klip može da prođe kroz praktično isti
`_load_vid` kao GRID primer i daje tenzor `(C, T, 64, 128)` normalizovan u `[0, 1]`.

### Faza 3 — Minimalni srpski Dataset adapter

**Cilj:** zadržati ugovor upstream `MyDataset` klase bez prethodnog manifest sloja.

- preuzeti `_load_vid`, `_padding`, `txt2arr`, `arr2txt`, `ctc_arr2txt`, `wer` i
  `cer` iz VIPL `dataset.py`;
- prilagoditi pronalaženje putanja lokalnoj `spkXX` strukturi;
- prilagoditi `_load_anno` tab-separated anotacijama, ukloniti `sil`/`sp` i
  normalizovati tekst;
- definisati srpski karakter skup neposredno uz `MyDataset.letters`;
- speaker-disjoint train/validation/test ID-jeve držati u maloj verzionisanoj
  konfiguraciji; Dataset pri pokretanju sam pronalazi pripadajuće uzorke;
- implementirati collate/padding samo koliko je potrebno za promenljive dužine;
- proveriti `target_length <= input_length` nakon vremenskog prolaza modela.

**Prihvatni kriterijum:** DataLoader vraća isti rečnik ključeva kao VIPL
(`vid`, `txt`, `txt_len`, `vid_len`), a dva različito duga klipa mogu u isti batch.

### Faza 4 — Srpski LipNet i transfer checkpoint-a

**Cilj:** promeniti samo ono što engleski checkpoint objektivno ne može da pokrije.

- parametrizovati broj klasa u preuzetom `LipNet` modelu;
- instancirati srpski head sa odgovarajućim brojem CTC klasa;
- učitati sve VIPL težine čiji se naziv i oblik slažu, kao u upstream `main.py`;
- eksplicitno prikazati da je preskočen samo izlazni `FC` sloj;
- zadržati konvolucione i oba BiGRU sloja;
- pokrenuti forward, CTC loss i jedan backward korak na malom batch-u.

**Prihvatni kriterijum:** nema neočekivano propuštenih backbone parametara, NaN
vrednosti niti CTC shape grešaka.

### Faza 5 — Baseline fine-tuning

**Cilj:** dobiti prvi srpski checkpoint adaptacijom VIPL trening petlje.

- zadržati upstream Adam, CTC loss, greedy decode, WER/CER i checkpoint tok;
- zameniti direktne `.cuda()` pozive jednim eksplicitnim `device` izborom;
- početi sa zamrznutim ili manjom stopom učenja za backbone, pa ga postepeno
  odmrznuti samo ako validacija to opravda;
- koristiti upstream horizontalni flip kao jedinu baseline augmentaciju;
- birati najbolji checkpoint prema validation WER-u;
- prijaviti i CER i sentence exact match, ali ne menjati kriterijum bez evidencije.

Baseline konfiguracija je zaključana na najviše 30 epoha i batch 2. Prve tri
epohe treniraju samo novi `FC` head sa stopom `1e-4`; zatim se ceo model odmrzava,
backbone koristi `2e-5`, a head ostaje na `1e-4`. Early stopping se aktivira posle
odmrzavanja i prekida nakon pet epoha bez strogo boljeg validation WER-a. Colab
čuva `latest.pt` za automatski nastavak i `best.pt` za završnu test evaluaciju.

**Prihvatni kriterijum:** sačuvan je reproduktivan checkpoint i inference radi na
potpuno neviđenim govornicima.

### Faza 6 — Traženi mali eksperimenti

**Cilj:** odgovoriti na zadatak bez menjanja osnovnog modela.

Na istom test splitu i sa istim baseline checkpoint-om uraditi:

1. originalni `128x64` mouth crop;
2. degradaciju `96x48 -> 128x64`;
3. degradaciju `64x32 -> 128x64`;
4. Gaussian blur;
5. clip-consistent crop jitter;
6. po potrebi jedan kratak nastavak fine-tuning-a sa blur/jitter augmentacijom.

Blur i crop jitter se dodaju kao male transformacije uz VIPL `cvtransforms.py`,
ne kao novi preprocessing sistem. Za svaki scenario prikazati CER, WER, sentence
exact match i nekoliko kvalitativnih predikcija.

**Prihvatni kriterijum:** svi scenariji razlikuju samo testiranu perturbaciju;
split, checkpoint i dekoder ostaju isti.

### Faza 7 — Jedan Colab notebook i finalni izveštaj

**Cilj:** omogućiti odbranu projekta iz jednog proverljivog toka.

Notebook treba da:

- pin-uje zavisnosti i prikazuje GPU;
- preuzima/učitava VIPL checkpoint;
- reprodukuje GRID inference;
- priprema mali AI-SPEAK uzorak istim preprocessing kodom;
- učitava najbolji srpski checkpoint i prikazuje predikciju;
- reprodukuje tabelu rezolucije, blur-a i jitter-a;
- jasno prikaže koje linije potiču iz VIPL repoa, a koje su nužne adaptacije.

Izveštaj objašnjava LipNet, CTC, GRID/AI-SPEAK razlike, rezultate, tipične greške,
ograničenja i sadržaj `docs/upstream-diff.md`.

## 7. Redosled zavisnosti

```text
pinovan VIPL commit
        ↓
originalni GRID inference
        ↓
VIPL demo preprocessing nad AI-SPEAK MP4
        ↓
minimalni srpski Dataset adapter
        ↓
srpski CTC head + transfer težina
        ↓
baseline fine-tuning
        ↓
rezolucija / blur / jitter evaluacije
        ↓
Colab demonstracija i izveštaj
```

Nije dozvoljeno paralelno razvijati lokalni trening sistem pre nego što su završeni
GRID parity test i AI-SPEAK preprocessing parity test.

## 8. Minimalni testovi

Testovi treba prvenstveno da potvrde kompatibilnost sa upstream-om:

- isti GRID frejmovi daju isti tensor shape i opseg vrednosti;
- isti engleski CTC niz daje isti tekst kao VIPL `ctc_arr2txt`;
- modernizovan model daje istu GRID predikciju kao pre izmene;
- srpski `txt2arr -> arr2txt` round-trip čuva sva slova;
- lokalni `.align` parser uklanja samo `sil`/`sp`;
- promenljivo dugi klipovi imaju ispravne `vid_len` i padding;
- checkpoint audit potvrđuje da je preskočen samo srpski head;
- jedan batch prolazi forward, CTC loss i backward.

Novi test nema vrednost ako proverava ponašanje koje je uvedeno samo starim
manifest/ROI pipeline-om.

## 9. Rizici i granice

| Rizik | Odgovor |
|---|---|
| Stari PyTorch i `face_alignment` API | Minimalna compatibility izmena uz parity test |
| AI-SPEAK klipovi nisu fiksnih 75 frejmova | Najmanja collate/padding adaptacija uz stvarne dužine |
| Full-frame MP4 nije GRID mouth crop | Ponovo koristiti poravnanje i crop iz VIPL `demo.py` |
| Srpski head nije kompatibilan sa checkpoint-om | Učitati backbone/GRU; novi head inicijalizovati istim pravilom |
| Mali skup i overfitting | Speaker-disjoint test, rano zaustavljanje i ograničen broj eksperimenata |
| Privatnost snimaka | MP4 i mouth frejmovi ostaju van Git-a |
| Upstream je teško reprodukovati | Prvo zaključati jedan GRID parity primer i tek zatim menjati kod |
| Nejasno poreklo koda | Pinovan SHA, MIT licenca i `upstream-diff.md` |

## 10. Definition of done

Projekat je završen kada:

- upstream commit i licenca su sačuvani;
- originalni VIPL GRID inference je reprodukovan;
- lokalni preprocessing koristi VIPL face-alignment i mouth-crop tok;
- trening ne zavisi od prethodnih statičnih manifesta i ROI artefakata;
- svako odstupanje od VIPL koda ima konkretan tehnički razlog i parity proveru;
- VIPL backbone i BiGRU težine su uspešno prenete u srpski model;
- postoji baseline checkpoint evaluiran na neviđenim govornicima;
- završeni su eksperimenti rezolucije, blur-a i crop jitter-a;
- prijavljeni su CER, WER, exact match i kvalitativne greške;
- jedan Colab notebook reprodukuje ključni tok za odbranu;
- privatni snimci, frejmovi i veliki checkpoint-i nisu poslati u Git.

## 11. Primarni izvori

- Assael et al., [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599)
- VIPL, [LipNet-PyTorch](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch)
- University of Sheffield, [GRID audiovisual sentence corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- Lokalni, od profesora dostavljen `processed.zip` AI-SPEAK podskup
