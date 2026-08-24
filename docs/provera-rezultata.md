# Provera sačuvanih rezultata notebookova 02–05

Datum audita: 24. avgust 2026.

## Zaključak

**Opšta ocena: potrebna je revizija pre citiranja završnih rezultata.**

Sačuvani izlazi potvrđuju da je praktični tok za Faze 2–5 zaista pokretan nad
AI-SPEAK podacima i GPU-om. Ipak, konačne metrike notebooka 05 nisu metodološki
uporedive i trening je koristio pogrešan tretman vremenskog padding-a. Stare
brojeve treba čuvati samo kao trag eksperimenta, ne kao rezultat za izveštaj ili
odbranu.

## Kontrolni izvori

- kod i izlazi sačuvani u notebookovima 02–05 pre ovog audita;
- checkpoint/rezultat putanje prikazane u tim notebookovima na Google Drive-u;
- lokalni testovi nad `lipnet/`, `scripts/` i generatorom notebookova;
- pinovani VIPL commit
  `40209e09c49553c00c25c7d41faa3706aea3c625`.

Privatni MP4/JPEG podaci i Drive checkpoint-i nisu dostupni u lokalnom
repozitorijumu, pa audit ne može nezavisno da rekonstruiše sve numeričke
rezultate bez ponovnog Colab pokretanja.

## Notebook 02 — preprocessing

Sačuvani izlaz je prikazivao:

- GPU: Tesla T4;
- 3.957 MP4 i 3.957 ALIGN fajlova;
- 186 vraćenih Drive checkpoint arhiva;
- 0 klipova u `failed_clips.log`;
- završnu arhivu `ai_speak_lip.zip` veličine približno 1,34 GiB.

Ovi brojevi su međusobno mogući, ali prethodni restore je vraćao JPEG fajlove
bez spajanja originalnih `decoded_frames`, `landmark_frames` i
`dropped_frames` zapisa. Za vraćene klipove je zato naknadno sintetički
prijavljivano `dropped_frames=0`, pa izbor „graničnih” QA klipova i tvrdnja o
ispuštenim frejmovima nisu pouzdani. Takođe nije postojao programski dokaz da je
ručni QA zaista potvrđen.

Ispravka sada vraća i spaja pravi JSONL audit iz svakog chunk-a, odbija delimične
foldere kao završene, proverava da uspešni i neuspešni zapisi tačno pokrivaju sve
ulaze i zahteva `MANUAL_QA_PASSED=True` pre arhiviranja.

**Status:** upotrebljivo uz caveat za broj uzoraka i postojanje arhive; ponovno
pokretanje Faze 2 je potrebno ako se u izveštaju navode dropped-frame/QA tvrdnje.

## Notebook 03 — Dataset i split

Sačuvani izlaz je prikazivao:

- 29 CTC klasa;
- split 16 train, 3 validation i 3 test govornika;
- 2.877 train, 540 validation i 540 test uzoraka (ukupno 3.957);
- primer batch-a oblika `(2, 3, 132, 64, 128)` sa `vid_len=[130,132]` i
  `txt_len=[27,30]`.

Zbir uzoraka se poklapa sa brojem parova iz Faze 2 i speaker liste su
disjunktne. Međutim, stari alfabet round-trip je koristio `.strip()`, čime je
iz testa nenamerno izbačen razmak. Novi notebook eksplicitno testira razmak i
proverava jednakost pronađenih mouth foldera i očekivanih anotacija.

**Status:** smer i broj uzoraka su konzistentni; ponoviti notebook posle
definitivnog Phase 2 artefakta radi završnog audita potpunosti.

## Notebook 04 — transfer i CTC smoke

Sačuvani izlaz je prikazivao:

- 22 preneta tenzora;
- namerno preskočene samo `FC.weight` i `FC.bias`;
- batch dužine 58 i 60 frejmova;
- logits `(2, 60, 29)`;
- konačan CTC loss `5.65675687789917` i uspešan backward.

Transfer audit je strukturno dobar. CTC loss je samo smoke vrednost, ne metrika
modela. Pošto je kraći uzorak tada prolazio kroz padded 3D CNN/BiGRU tok, tačna
loss vrednost nije uporediva sa ispravljenim length-aware modelom.

**Status:** transfer težina je dobro potkrepljen; GPU smoke treba ponoviti sa
novim modelom.

## Notebook 05 — trening i završne metrike

U starom notebooku su zabeleženi sledeći brojevi:

- strong fine-tuning: najbolja epoha 66, validation WER
  `0.5200617283950617`;
- greedy test: WER `0.47222222222222215`, CER
  `0.16367445495476837`, sentence exact match `0.0`, 540 uzoraka;
- character-LM test: WER `0.4351851851851852`, CER
  `0.1587747287811104`.

Ove vrednosti se **ne smeju citirati kao potvrđeni rezultat**, iz sledećih
razloga:

1. Bidirekcioni GRU je obrađivao padded vremenske korake. Dodatno, vremenske 3D
   konvolucije su širile signal u padding, pa je kraći klip davao drugačije
   validne logite kada je bio u batch-u sa dužim klipom.
2. Notebook nije izvršen odozgo nadole. Execution count pokazuje da je strong
   setup pokrenut pre baseline trening ćelije, a postojeći `latest.pt` je već bio
   na epohi 30.
3. Baseline log je zbog duplih vitičastih zagrada štampao literalni tekst
   `{epoch...}` umesto brojeva.
4. Strong trening nije imao pouzdan resume, RNG/history checkpoint ni odvojenu
   verziju protokola; ponovnim pokretanjem počinjao je iznova i mogao da prepiše
   rezultate.
5. Greedy CER je bio prosečna sentence-normalized vrednost, dok je LM CER bio
   corpus zbir grešaka podeljen zbirom karaktera. Njihova razlika zato nije
   validno poboljšanje.
6. Greedy baseline vrednosti za LM poređenje bile su ručno upisane, uključujući
   zaokružen `CER=0.1637`, umesto ponovo izračunate iz istog cache-a.
7. Ručno dodat strong/LM nastavak nije postojao u generatoru notebooka, pa Git
   nije imao jedan reproduktivan izvor.

Stari character-LM dodatak je uklonjen iz baseline notebooka. Novi Faza 5 tok
koristi length-aware model, jednu corpus-level definiciju metrika, novi Drive
folder `phase5_length_aware_v2` i checkpoint hash pri keširanju test rezultata.

**Status:** potrebna potpuna GPU revizija; stare metrike su samo istorijski
zabeležene vrednosti.

## Obavezni redosled ponovne validacije

1. Pokrenuti Fazu 02 ako su potrebne pouzdane dropped-frame i QA tvrdnje.
2. Pokrenuti Fazu 03 nad konačnom `ai_speak_lip.zip` arhivom.
3. Pokrenuti Fazu 04 i sačuvati novi transfer/CTC audit.
4. Pokrenuti Fazu 05 od početka u novom `phase5_length_aware_v2` folderu.
5. Pokrenuti Fazu 06. Njena baseline evaluacija mora se bit-po-metrici poklopiti
   sa Phase 5 `results.json` pre nego što se prihvate rezultati rezolucije,
   blur-a i crop pomeranja.

Za završni izveštaj navesti checkpoint SHA-256, broj test uzoraka, definiciju
metrika (`corpus-edit-distance-v1`) i rezultate iz novog
`robustness_results.json` fajla.
