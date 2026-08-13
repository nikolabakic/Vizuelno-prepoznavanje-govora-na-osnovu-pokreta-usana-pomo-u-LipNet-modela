# Analiza praktičnog dela i roadmap

Datum analize: 13. avgust 2026.

## 1. Sažetak odluke

Praktični deo treba da ostane mali i razumljiv studentski projekat. Predloženi put je:

1. Pokrenuti originalni, predtrenirani engleski LipNet na nekoliko GRID primera i potvrditi ceo inference pipeline.
2. Napraviti pouzdan manifest lokalnog srpskog korpusa, speaker-disjoint podelu i stabilan crop regiona usana.
3. Preuzeti predtrenirane prostorne i vremenske slojeve, zameniti izlazni sloj srpskim alfabetom i fino podesiti model na AI-SPEAK podskupu.
4. Uraditi samo dva trening scenarija: osnovni fine-tuning i kratki nastavak treninga sa blagim blur/crop-jitter augmentacijama.
5. Na istom test skupu izmeriti uticaj rezolucije, blur-a i crop jitter-a; prikazati WER, CER, exact-match i kvalitativne greške.
6. Koristiti isključivo greedy CTC dekodiranje kao u VIPL LipNet-PyTorch evaluaciji, bez jezičkog modela i gramatičkog dekodera.

Ovo pokriva sve tačke praktičnog zadatka bez treniranja LipNet-a od nule i bez nepotrebno velikog testnog ili infrastrukturnog sloja.

Oznake resursa u dokumentu:

- **CPU** — lokalni računar ili Colab bez GPU-a je dovoljan.
- **GPU preporučen** — CPU može za mali smoke test, ali bi rad bio prespor.
- **GPU obavezan** — plan nije praktično izvodljiv u razumnom studentskom roku bez GPU-a.

## 2. Pokrivenost zahteva praktičnog dela

| Zahtev | Planirani rezultat |
|---|---|
| Inference postojećim LipNet modelom | Predtrenirani VIPL model na 3–5 GRID video primera, sa prikazom ground-truth i predikcije |
| Analiza učitavanja i pripreme frejmova | Manifest, dekodiranje MP4, izbor frejmova, detekcija usana, crop, resize, normalizacija i temporalno padding/bucketing |
| Promena rezolucije | Evaluacija `128×64`, `96×48 → 128×64` i `64×32 → 128×64` na istom checkpoint-u |
| Blur i crop jitter | Kontrolisane test-korupcije i jedan kratki augmentovani fine-tuning |
| Fine-tuning na delu AI-SPEAK baze | Transfer učenja sa engleskog LipNet-a na srpski izlazni alfabet |
| Kvalitativni rezultati | Tabela tačnih/netačnih primera, tip greške i komentar o mogućem uzroku |
| Finalni izveštaj | Arhitektura i CTC, tabela eksperimenata, WER/CER, kvalitativna analiza i ograničenja |
| Usmena odbrana | Jedan Colab notebook koji učitava checkpoint, pokreće inference i reprodukuje ključne grafikone |

## 3. Stanje repozitorijuma

Pre ove analize repozitorijum je bio generički Python šablon: `app/`, `docs/`, `docker/` i `playground/` nisu sadržali programski kod, a `pyproject.toml` nije imao ML zavisnosti. Tekst zadatka upućuje na:

- rad [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599),
- [GRID audiovisual sentence corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/),
- [VIPL LipNet-PyTorch implementaciju](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch).

Lokalni korpus i ZIP arhiva bili su neversionisani, ali ne i ignorisani. Dodati su u `.gitignore`, zajedno sa checkpoint-ima i trening artefaktima. To sprečava slučajno slanje više gigabajta podataka i prepoznatljivih lica na javni GitHub.

## 4. Audit lokalnog `processed/` korpusa

Ovo su svojstva stvarnih fajlova dostupnih u projektu, a ne samo opis zvanične baze.

### 4.1. Struktura i obim

```text
processed/processed/
├── mappings.txt                 # postoji, ali je prazan
├── spk01/
│   ├── alignment/*.align
│   └── ser/video_a/*.mp4
├── ...
└── spk22/
```

| Svojstvo | Lokalni nalaz |
|---|---:|
| Govornici | 22 (`spk01`–`spk22`) |
| MP4 snimci | 3.957 |
| `.align` anotacije | 3.957 |
| Neupareni video/anotacija fajlovi | 0 |
| Snimci po govorniku | uglavnom 180; `spk09=179`, `spk15=178` |
| Veličina video fajlova | 3,806 GiB |
| Ukupno trajanje videa | 5,90 h |
| Trajanje jednog videa | min 2,32 s; prosek 5,368 s; P95 7,52 s; max 13,12 s |
| Broj frejmova | min 57; prosek 132,7; P95 185; max 325 |
| Video format | MP4, AVC/H.264, `1280×720`, 25 fps |
| Audio unutar MP4 | postoji AAC audio tok, iako će model koristiti samo sliku |
| Jezik/pogled | samo `ser/video_a` u ovoj izvedenoj verziji |

Rezolucija, kodek i frame rate provereni su na po tri snimka svakog govornika; svih 66 proverenih fajlova imalo je ista svojstva. Trajanje i broj frejmova očitani su iz svih 3.957 MP4 kontejnera.

### 4.2. Anotacije

`.align` je UTF-8 tekst sa tri tabulatorom odvojene kolone:

```text
početak    kraj    token
```

Primer sadrži početni i završni `sil` i tačno šest govornih tokena. Lokalni vremenski brojevi odgovaraju koraku od `10^-7` sekundi (na primer, `50.000.000 ≈ 5 s`), a ne milisekundama iz opisa izvorne baze. Anotacije ukupno pokrivaju 5,74 h, dok MP4 tokovi traju 5,90 h; video je u proseku oko 0,145 s duži od poslednje oznake. Između susednih reči često postoje neoznačene pauze; zabeleženo je 20.477 takvih razmaka. To nije problem za CTC jer će se iz anotacije koristiti samo redosled teksta, ali jeste važno ako se kasnije koriste word-level vremenske granice.

Sve rečenice imaju šest pozicija:

| Pozicija | Kategorija | Broj izbora | Primeri |
|---:|---|---:|---|
| 1 | komanda | 7 | `dalje`, `kraj`, `obriši`, `odustani`, `početak`, `pošalji`, `potvrdi` |
| 2 | slovo | 30 | srpska latinična slova, uključujući `č`, `ć`, `đ`, `dž`, `lj`, `nj`, `š`, `ž` |
| 3 | smer | 6 | `desno`, `dole`, `gore`, `levo`, `napred`, `nazad` |
| 4 | slovo | 30 | isti skup kao pozicija 2 |
| 5 | dan | 7 | `ponedeljak`–`nedelja` |
| 6 | cifra | 10 | `nula`–`devet` |

Teorijski postoji `7 × 30 × 6 × 30 × 7 × 10 = 2.646.000` dozvoljenih kombinacija. Lokalno postoje 3.955 jedinstvene transkripcije među 3.957 primera. Ukupan word-level vokabular ima 60 tokena.

Karakter-level target koristi 27 vidljivih slova plus razmak:

```text
a b c č ć d đ e f g h i j k l m n o p r s š t u v z ž [space]
```

`dž`, `lj` i `nj` ostaju nizovi od dva karaktera, što je u redu za karakter-level CTC. Sa CTC blank simbolom model mora imati **29 izlaznih klasa**. Prazan `mappings.txt` ne sme se koristiti; deterministički `vocab.json` treba generisati iz trening konfiguracije i čuvati uz checkpoint.

### 4.3. Vizuelni sadržaj

Pregled uzoraka pokazuje:

- pun, prepoznatljiv kadar lica, a ne gotov crop usana;
- uglavnom frontalnu pozu i kontrolisano osvetljenje;
- promene razmere i vertikalnog položaja lica;
- različite pozadine i odeću;
- naočare, refleksije na staklima, bradu/brkove i druge realne smetnje.

Zbog toga je obavezna stabilna ekstrakcija regiona usana. Direktno smanjivanje celog `1280×720` kadra na `128×64` odbacilo bi gotovo sve korisne detalje usana.

### 4.4. Privatnost i način korišćenja

`processed.zip` je profesor obezbedio kao ulazni korpus za ovaj projekat. Preuzimanje, formiranje i rekonstrukcija korpusa zato nisu deo praktičnog rada. Koristi se upravo dobijena lokalna verzija iz direktorijuma `processed/`.

Arhiva i raspakovani podaci ostaju lokalno i izvan Git-a. Pošto snimci sadrže puna, prepoznatljiva lica, podatke, cele frejmove i njihove screenshot-ove ne treba objavljivati. Za izveštaj koristiti samo rezultate, agregirane metrike i, ako je potrebno i dozvoljeno, izdvojene crop-ove usana.

## 5. Zvanični AI-SPEAK i lokalna izvedena verzija

Zvanični [AI-SPEAK opis baze](https://data.telekom.ftn.uns.ac.rs/ai-speak/catA/ai-speak%20database.pdf) iz 2026. i [javni Category A indeks](https://data.telekom.ftn.uns.ac.rs/ai-speak/catA/) navode 30 govornika, 15 žena i 15 muškaraca. Svaki govornik ima 160 snimaka: 80 srpskih i 80 engleskih. Baza uključuje frontalnu kameru, dve mobilne kamere pod približno ±30° i profesionalni mikrofon. Frontalni video je 100 fps, bočne kamere 30 fps, a zasebni mono WAV je 22,05 kHz. Javna verzija je anonimizovana i licencirana kao CC BY-NC-SA 4.0.

Rad „Snimanje bilingvalne baze AI-SPEAK za multimodalno prepoznavanje govora“ u [YUINFO 2025 zborniku](https://www.yuinfo.org/zbornici/2025/YUINFO2025.pdf) opisuje tadašnju fazu sa 25 govornika. Razlika 25 → 30 najverovatnije predstavlja razvoj korpusa između rada iz 2025. i javnog izdanja iz 2026.

| Svojstvo | Zvanično javno izdanje | Lokalni `processed/` |
|---|---|---|
| Govornici | 30 | 22 |
| Jezici | srpski + engleski | samo srpski |
| Primeri po govorniku | 80 po jeziku | oko 180 srpskih |
| Pogledi | frontalni + levi + desni | samo `video_a` |
| Frontalni fps | 100 fps | 25 fps |
| Audio | zaseban WAV 22,05 kHz | AAC unutar MP4 |
| Privatnost slike | lip-only, okolina pikselizovana/maskirana | puno prepoznatljivo lice |
| Anotacija | automatski word alignment; moguća odstupanja | transformisani `.align`, šest tokena, jedinice od `10^-7 s` |
| Organizacija | 80 zajedničkih/personalizovanih rečenica po jeziku | GRID-slična šestopoziciona gramatika |

Zaključak: lokalni skup treba tretirati kao zaseban, izvedeni dataset. Svojstva originalnog AI-SPEAK izdanja ne smeju se automatski pripisati lokalnim fajlovima.

## 6. Poređenje sa GRID korpusom

Zvanični [GRID sajt](https://spandh.dcs.shef.ac.uk/gridcorpus/) navodi 34 govornika i po 1.000 rečenica; video govornika 21 nije dostupan. LipNet rad navodi 32.746 upotrebljivih videa i oko 28 sati. Svi LipNet eksperimenti koriste klipove dužine 3 s na 25 fps i mouth-centred crop `100×50`; VIPL reprodukcija koristi `128×64`.

| Svojstvo | GRID korišćen u LipNet radu | Lokalni srpski korpus |
|---|---|---|
| Jezik | engleski | srpski, latinica |
| Govornici | 34 navedena; jedan video skup nedostaje | 22 |
| Upotrebljivi primeri | 32.746 | 3.957 |
| Ukupno trajanje | oko 28 h | 5,90 h videa |
| Klip | fiksno 3 s / 75 frejmova | 2,32–13,12 s / 57–325 frejmova |
| Rečenica | tačno 6 reči | tačno 6 tokena |
| Gramatika | `4×4×4×25×10×4 = 64.000` kombinacija | `7×30×6×30×7×10 = 2.646.000` kombinacija |
| Word vokabular | oko 51 reč/token | 60 tokena |
| Model alfabet | 26 engleskih slova + razmak + blank = 28 izlaza | 27 srpskih karaktera + razmak + blank = 29 izlaza |
| Ulaz u LipNet | već poravnat crop usana | pun HD kadar; crop tek treba napraviti |
| Evaluacija | unseen-speaker i overlapped-speaker | primarno mora biti unseen-speaker |

Najvažnije posledice su:

- lokalno imamo oko osam puta manje primera i skoro pet puta manje sati;
- izlazni klasifikator predtreniranog modela nije kompatibilan sa srpskim alfabetom;
- lokalne sekvence su znatno duže i promenljive, pa originalni `vid_padding=75` nije validan;
- ROI domen nije isti: predtrenirani model očekuje stabilizovan crop usana, ne celo lice;
- veoma veliki prostor mogućih rečenica znači da model ne može „zapamtiti“ sve kombinacije, iako mu fiksna gramatika mnogo pomaže;
- speaker-overlap podela bi dala optimističan rezultat; glavni rezultat mora biti na potpuno neviđenim govornicima.

## 7. Analiza LipNet rada i GitHub implementacije

Originalni LipNet je:

```text
video usana → 3 × (3D konvolucija + spatial pooling) → 2 × BiGRU
            → linearni sloj po vremenskom koraku → CTC
```

Model radi na karakterima i ne zahteva ručno poravnanje svakog karaktera sa frejmom. CTC marginalizuje dozvoljena poravnanja i uklanja ponovljene karaktere i blank simbole.

[VIPL LipNet-PyTorch](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch) je dobar referentni početak jer sadrži dva predtrenirana checkpoint-a i prijavljuje 13,3% WER na unseen-speaker i 4,6% WER na overlapped-speaker GRID podeli. Ipak, kod nije spreman za direktnu upotrebu na našem skupu:

- zasnovan je na PyTorch 1.0+ stilu i svuda direktno poziva `.cuda()`;
- očekuje direktorijume JPEG frejmova, ne naše MP4 fajlove;
- očekuje tačno `128×64` i fiksni vremenski padding 75;
- koristi samo engleski alfabet i izlazni sloj od 28 klasa;
- demo koristi stariji `face_alignment` API i shell pozive ka FFmpeg-u;
- trening koristi `DataParallel`, veliki batch i fiksne putanje;
- repozitorijumska evaluacija koristi greedy argmax dekodiranje; isti pristup koristiće i ovaj projekat;
- sami autori reprodukcije upozoravaju da je rezultate teško reprodukovati.

Plan je zato da se zadrže arhitektura i kompatibilni predtrenirani parametri, ali napiše mali savremeni loader/trainer. Kopiranje celog starog repozitorijuma i njegovo nasumično krpljenje bilo bi manje pregledno za odbranu.

## 8. Da li LipNet koristi LLM/SLM?

Ne u današnjem značenju tih termina. Osnovni LipNet nije veliki jezički model. To je vizuelni encoder + rekurentni sekvencijalni model + CTC.

Originalni rad je za konačne WER/CER rezultate koristio CTC beam search sa **karakterskim 5-gramskim jezičkim modelom**, ali VIPL LipNet-PyTorch evaluacija koju reprodukujemo koristi greedy argmax CTC dekodiranje.

### Odluka za srpski

Projekat koristi isključivo **karakterski LipNet, CTC loss i greedy CTC dekodiranje**, kao VIPL LipNet-PyTorch evaluacija. Za svaki vremenski korak bira se najverovatnija klasa, zatim se uklanjaju uzastopna ponavljanja i CTC blank simboli.

Ne implementiraju se beam search, n-gram ili drugi jezički model, grammar-constrained dekodiranje, N-best reranking niti GPT/BERT dekoder. Fiksna šestopoziciona struktura korpusa koristi se samo za opis podataka i analizu tačnosti po pozicijama; ne koristi se za ispravljanje ili ograničavanje predikcije.

## 9. Predloženi tehnički pipeline

```text
MP4 + .align
    ↓
manifest i speaker-disjoint split
    ↓
dekodiranje frejmova (25 fps)
    ↓
detekcija/poravnanje lica i stabilan mouth ROI
    ↓
RGB crop 128×64 + normalizacija
    ↓
batch po sličnim dužinama + dinamički padding
    ↓
LipNet backbone + novi srpski linearni sloj (29 klasa)
    ↓
CTC loss
    ↓
greedy CTC predikcija
    ↓
CER / WER / exact match + kvalitativna analiza
```

### ROI i keširanje

Za jednostavan projekat preporučuje se MediaPipe/face-landmark detekcija na nekoliko reprezentativnih frejmova svakog videa, izračunavanje jednog stabilnog mouth ROI-a za ceo klip i mali padding oko usana. Time se izbegava jitter od detekcije svakog frejma i višesatna ponovna obrada pri svakoj epohi.

- Prvi prolaz detekcije i pravljenje ROI manifesta: **CPU**.
- Ako CPU detekcija postane usko grlo, `face_alignment` na Colab GPU-u je rezervna opcija: **GPU preporučen**.
- Trening ne sme ponovo pokretati landmark detektor; koristi se zapamćen ROI ili prethodno pripremljeni mouth-only klipovi.

Pre masovne obrade ručno pregledati mrežu crop-ova za najmanje 30 videa i sve govornike. Potreban je fallback na poslednji validan/medijanski ROI ako detekcija omane.

### Promenljiva dužina

Originalni padding od 75 frejmova ne radi za lokalni skup. Potrebno je:

- čuvati stvarni broj frejmova;
- grupisati primere slične dužine u batch-eve;
- padovati samo do najdužeg primera u batch-u;
- koristiti stvarne `input_lengths` za CTC;
- koristiti packed sequences ili ekvivalent kako BiGRU ne bi učio iz padding-a;
- ograničiti ekstremno dugačke klipove tek posle pregleda, ne automatski odseći govor.

### Eksperiment rezolucije

VIPL model ima dimenziju GRU ulaza vezanu za `128×64`. Da promena rezolucije ne bi istovremeno menjala arhitekturu, čist test je:

```text
mouth crop 128×64
  ├─ ostaje 128×64                 # baseline
  ├─ downsample 96×48 → 128×64     # srednja degradacija
  └─ downsample 64×32 → 128×64     # jaka degradacija
```

Tako merimo gubitak vizuelnog detalja, a ne razliku u broju parametara.

## 10. Podela podataka

Primarni eksperiment mora biti **speaker-disjoint**:

- oko 16 govornika za trening,
- 3 za validaciju,
- 3 za test,
- fiksan seed i trajno sačuvan `split.json`.

Tačni ID-jevi se biraju deterministički nakon što se potvrdi da metadata ne sadrži preporučenu zvaničnu podelu. Ne treba deliti nasumične video snimke istog govornika između train/test skupa, jer bi model mogao da koristi identitet, bradu, naočare, pozadinu i geometriju lica umesto da generalizuje pokrete usana.

Mali overlapped-speaker rezultat može postojati samo kao sekundarna demonstracija poređenja sa GRID literaturom i mora biti jasno označen kao lakši scenario.

## 11. Minimalna matrica eksperimenata

Plan ograničava skupe treninge na najviše dva.

| ID | Model/trening | Evaluacija | Resurs |
|---|---|---|---|
| E0 | Originalni GRID checkpoint, bez treninga | 3–5 GRID primera | GPU preporučen; CPU moguć za jedan primer |
| E1 | Srpski fine-tuning bez blur/jitter augmentacije | čist speaker-disjoint test | **GPU obavezan** |
| E2 | E1 checkpoint | `128×64`, `96×48`, `64×32` | GPU preporučen |
| E3 | E1 checkpoint | čist, Gaussian blur i crop jitter | GPU preporučen |
| E4 | Kratak nastavak E1 sa blagim blur + jitter augmentacijama | isti čist i korumpiran test | **GPU obavezan** |

Predložene kontrolisane perturbacije:

- Gaussian blur: kernel `5×5`, približno `sigma=1.0–1.2`;
- crop jitter: zajednički pomak celog klipa do približno `±4 px`, bez nezavisnog skakanja svakog frejma;
- horizontalni flip može biti trening augmentacija, ali ne zamenjuje tražene blur/jitter eksperimente.

Za svaki red prikazati najmanje CER, WER i sentence exact-match. Korisno je dodati tačnost po jednoj od šest pozicija, jer odmah pokazuje da li model najviše greši slova, smerove, dane ili cifre.

## 12. Roadmap

> **Polazno stanje:** profesor je obezbedio `processed.zip` i u svim narednim fazama koristi se isključivo njegov lokalni sadržaj iz direktorijuma `processed/`. Arhiva i raspakovani podaci ostaju van Git-a. Preuzimanje ili priprema AI-SPEAK baze nisu deo roadmapa; rad počinje Fazom 1.

### Faza 1 — Originalni LipNet inference

**Resurs: GPU preporučen; CPU je prihvatljiv za jedan smoke test. Procena: jedan dan.**

- preuzeti unseen-speaker VIPL checkpoint;
- obezbediti 3–5 legalno preuzetih GRID primera i anotacija;
- reprodukovati mouth crop i normalizaciju očekivanu checkpoint-om;
- prikazati video/crop, ground truth, predikciju, CER i WER;
- potvrditi da pipeline radi pre srpske adaptacije.

Kriterijum završetka: inference iz svežeg Colab runtime-a završava bez ručnih izmena putanja.

### Faza 2 — Manifest, QA, split i ROI

**Resurs: CPU; GPU samo ako se koristi teži landmark model. Procena: 1–2 dana plus vreme obrade.**

- napraviti `manifest.csv` isključivo iz lokalnog `processed/` korpusa, sa speaker ID-jem, video/anotacija putanjom, transkriptom, trajanjem i brojem frejmova;
- generisati `vocab.json` i `split.json`;
- implementirati stabilan mouth ROI i fallback;
- pregledati najmanje 30 crop-ova, sve govornike i nekoliko najdužih klipova;
- keširati samo ono što realno ubrzava trening.

Kriterijum završetka: svi primeri su upareni, split nema zajedničke govornike i nijedan target ne krši CTC uslov dužine.

### Faza 3 — Adaptacija modela na srpski

**Resurs: GPU obavezan. Procena: 1–2 dana rada; stvarni trening nekoliko sati na Colab T4 zavisno od batch-a.**

- učitati sve kompatibilne predtrenirane slojeve, ali srpski fine-tuning obaviti isključivo na lokalnom `processed/` korpusu;
- zameniti engleski linearni head slojem od 29 klasa;
- prvo kratko trenirati novi head i vremenski deo uz zamrznute 3D konvolucije;
- zatim odmrznuti ceo model uz manji learning rate;
- koristiti mixed precision, mali batch (`4–8` kao početna proba), gradient clipping i early stopping po validation WER-u;
- čuvati najbolji checkpoint, ne svaki epoch.

Kriterijum završetka: E1 daje smislene srpske predikcije na neviđenim govornicima i bolji je od trivijalnog/nenaučenog baseline-a.

### Faza 4 — Rezolucija i augmentacije

**Resurs: GPU preporučen za evaluaciju, GPU obavezan samo za E4 nastavak treninga. Procena: jedan dan.**

- zamrznuti test skup i sve perturbacije;
- pokrenuti E2 i E3 bez menjanja checkpoint-a;
- kratko nastaviti trening sa blur + jitter augmentacijama;
- ponoviti potpuno istu evaluaciju za E4;
- sačuvati tabelu rezultata i nekoliko reprezentativnih crop/prediction primera.

Kriterijum završetka: postoji jedna porediva tabela iz koje se vidi cena niže rezolucije i da li augmentacije popravljaju robustnost.

### Faza 5 — Greedy CTC evaluacija i analiza grešaka

**Resurs: GPU za brzo generisanje logita; greedy dekodiranje i analiza mogu na CPU-u. Procena: pola dana.**

- dekodirati izlaz najboljeg modela isključivo greedy CTC postupkom;
- prijaviti CER, WER i sentence exact-match na speaker-disjoint test skupu;
- izračunati tačnost za svaku od šest pozicija;
- analizirati supstitucije, brisanja i umetanja, naročito vizuelno slične glasove i slova;
- sačuvati reprezentativne predikcije i greške za izveštaj.

Kriterijum završetka: greedy rezultat je reproduktivan i kompletno analiziran bez jezičkog modela ili gramatičkog ograničavanja.

### Faza 6 — Izveštaj, notebook i odbrana

**Resurs: CPU. Procena: 1–2 dana.**

- finalizovati Colab notebook za demo;
- uneti arhitekturu i kratko objašnjenje CTC-a iz teorijskog dela;
- prikazati tabelu E0–E4, krive treninga i kvalitativne primere;
- objasniti razliku GRID/AI-SPEAK, speaker split, rizike i ograničenja;
- pripremiti kratak tok usmene demonstracije.

Kriterijum završetka: druga osoba može iz README-ja pokrenuti inference, a svi brojevi u izveštaju mogu se reprodukovati iz sačuvanih konfiguracija.

## 13. Minimalni testovi

Ne treba praviti veliki testni framework. Dovoljna su četiri automatizovana testa i nekoliko vizuelnih QA provera:

1. **Manifest test:** video i `.align` postoje, ID-jevi se slažu, transkript nije prazan.
2. **Split test:** skupovi govornika za train/validation/test su disjunktni.
3. **Vocab/CTC test:** svaki karakter je u `vocab.json`, `target_length ≤ input_length`, blank ID je konzistentan.
4. **Model smoke test:** jedan mali batch prolazi forward, CTC loss i jedan backward korak bez NaN/shape greške.

Vizuelno, ručno proveriti ROI, augmentacije i nekoliko dekodiranih rečenica. To je vrednije od desetina sitnih unit testova za notebook kod.

## 14. Glavni rizici i mitigacije

| Rizik | Posledica | Mitigacija |
|---|---|---|
| Lokalni korpus ne sme u Git | slučajno objavljivanje više gigabajta podataka i snimaka lica | zadržati `/processed/` i `/processed.zip` u `.gitignore`; verzionisati samo manifest/metapodatke bez kadrova |
| Prepoznatljiva lica | curenje ličnih podataka | ne objavljivati cele frejmove; koristiti dozvoljene mouth crop-ove |
| Prazan `mappings.txt` | pogrešan broj/raspored klasa | generisati i verzionisati `vocab.json`; 29 izlaza sa blank-om |
| Samo 3.957 primera | brzo overfitting | transfer learning, speaker split, early stopping, blage augmentacije |
| Pun HD kadar umesto mouth ROI-a | checkpoint ne vidi očekivani domen | landmark crop, vremenska stabilizacija i ručni QA |
| Promenljivi klipovi do 325 frejmova | OOM i mnogo padding-a | length bucketing, dinamički padding, mali batch, mixed precision |
| BiGRU obrađuje padding | loš backward kontekst | packed sequences ili maskirano/validno temporalno procesiranje |
| Automatske anotacije i praznine | pogrešni targeti/granice | CTC koristi redosled teksta; ručno pregledati anomalije i outliere |
| Speaker leakage | nerealno dobar rezultat | primarni test isključivo na neviđenim govornicima |
| Stari VIPL kod | problemi sa novim PyTorch/CUDA/Colab verzijama | zadržati arhitekturu/težine, napisati mali savremeni loader i trainer |
| English → Serbian transfer | izlazni head i neki vizuelni obrasci nisu kompatibilni | zameniti head; staged unfreezing; ne očekivati GRID WER |
| Rezolucija menja i arhitekturu | nepošten eksperiment | downsample pa upsample na stalnih `128×64` |
| Colab/Drive I/O | GPU čeka podatke | lokalni runtime storage, umeren broj keš fajlova, `num_workers` izmeriti |
| Previše trening varijanti | projekat postaje skup i nepregledan | samo E1 i E4 treniraju; ostalo su evaluacije istog checkpoint-a |

## 15. Predložena jednostavna struktura koda

```text
app/
├── data.py          # manifest, anotacije, split, Dataset/Collate
├── preprocessing.py # video decode, ROI, resize i augmentacije
├── model.py         # prilagođeni LipNet
├── train.py         # fine-tuning i checkpoint
├── evaluate.py      # CER/WER, eksperimenti i kvalitativni izlaz
└── decode.py        # isključivo greedy CTC dekodiranje
playground/
└── lipnet_colab.ipynb
docs/
└── analiza-i-roadmap.md
```

Konfiguracija može biti jedan mali YAML/JSON ili argumenti komandne linije. Nisu potrebni servis, baza podataka, Docker deployment ni web aplikacija za ispunjenje zadatka.

## 16. Definition of done

Projekat je završen kada:

- postojeći LipNet checkpoint uspešno radi inference na GRID primeru;
- srpski manifest, vokabular i speaker-disjoint split su sačuvani i provereni;
- mouth ROI pipeline je vizuelno validiran;
- postoji najbolji srpski fine-tuned checkpoint;
- urađeni su eksperimenti rezolucije, blur-a i crop jitter-a;
- prijavljeni su CER, WER i kvalitativni primeri na neviđenim govornicima;
- rezultat je dobijen isključivo greedy CTC dekodiranjem, bez LM-a ili gramatičkog ograničavanja;
- Colab notebook može da demonstrira inference;
- finalni izveštaj sadrži arhitekturu, CTC, rezultate, tipične greške, GRID/AI-SPEAK razlike i ograničenja;
- dataset, puni frejmovi i veliki checkpoint-i nisu slučajno objavljeni u Git-u.

## 17. Primarni izvori

- Assael et al., [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599)
- University of Sheffield, [GRID audiovisual sentence corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- VIPL, [LipNet-PyTorch](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch)
- AI-SPEAK, [zvanični opis javne baze](https://data.telekom.ftn.uns.ac.rs/ai-speak/catA/ai-speak%20database.pdf)
- AI-SPEAK, [Category A javni fajlovi](https://data.telekom.ftn.uns.ac.rs/ai-speak/catA/)
- Suzić et al., [Snimanje bilingvalne baze AI-SPEAK za multimodalno prepoznavanje govora](https://www.yuinfo.org/zbornici/2025/YUINFO2025.pdf)
