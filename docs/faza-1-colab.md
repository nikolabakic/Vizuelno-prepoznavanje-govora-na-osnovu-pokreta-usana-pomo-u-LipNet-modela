# Faza 1: originalni LipNet inference u Google Colab-u

Ova faza proverava da li originalni engleski LipNet pipeline radi od početka do kraja:

```text
GRID video -> 68 face landmarks -> poravnanje lica -> mouth ROI 128x64
           -> LipNet checkpoint -> greedy CTC -> tekst -> CER/WER
```

Srpski `processed/` skup se u ovoj fazi **ne koristi za inference**. On ima puni HD kadar,
promenljivu dužinu i srpski alfabet, dok originalni checkpoint očekuje GRID mouth crop i
28 engleskih izlaznih klasa. `processed/` počinje da se koristi u Fazi 2 (manifest i ROI),
a zatim u Fazi 3 (srpski fine-tuning).

## Šta je potrebno

- Google nalog i Google Colab;
- Colab runtime sa GPU-om (T4 je dovoljan);
- internet u Colab runtime-u radi preuzimanja paketa, checkpoint-a i GRID demo snimka;
- oko 1 GB slobodnog privremenog prostora;
- `processed/` na Google Drive-u nije potreban za Fazu 1, ali notebook može da proveri da li
  ga vidi na očekivanoj putanji.

Ne treba instalirati CUDA-u ručno. Colab već daje PyTorch i CUDA runtime.

## Pokretanje

1. Otvori [lipnet_faza1_colab.ipynb](../playground/lipnet_faza1_colab.ipynb) u Colab-u
   (`Open in Colab` sa GitHub-a ili upload notebook-a preko `File -> Upload notebook`).
2. Izaberi `Runtime -> Change runtime type -> T4 GPU -> Save`.
3. Pokreni ćelije redom preko `Runtime -> Run all`.
4. Ako Colab traži restart posle instalacije paketa, uradi restart i ponovo `Run all`.
5. Na kraju sačuvaj ispis: naziv GPU-a, broj učitanih parametara, ground truth, predikciju,
   CER i WER. To je dokaz da je smoke test završen.

Prvo pokretanje obično traje nekoliko minuta jer `face-alignment` preuzima svoj landmark
model i obrađuje frejmove. Sam LipNet inference traje kratko.

## Gde treba da bude `processed/`

Ako je raspakovan direktno na Drive-u, tipične putanje su:

```text
/content/drive/MyDrive/processed/processed/spk01/...
/content/drive/MyDrive/processed/spk01/...
```

Notebook montira Drive samo u opcionoj poslednjoj ćeliji i automatski proverava obe varijante.
Ako je folder negde drugde, promeni samo vrednost `PROCESSED_ROOT` u toj ćeliji. Nemoj kopirati
ceo skup u `/content` za Fazu 1.

## Očekivani problemi

- `CUDA nije dostupna`: GPU nije uključen; promeni runtime type na T4 GPU.
- `GRID video nije moguće otvoriti`: ponovo pokreni ćeliju za download; server ponekad prekine vezu.
- landmark model ne pronađe lice: ponovo pokreni preprocessing; notebook prijavljuje tačno koji
  frejm nije mogao da interpolira.
- predikcija nije savršena: to samo po sebi nije kvar. Bitno je da je checkpoint potpuno učitan,
  ROI smislen i da inference daje tekst; prijavljeni rezultat checkpoint-a odnosi se na ceo test skup.

Izvori: [VIPL LipNet-PyTorch](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch)
i [zvanični GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/).
