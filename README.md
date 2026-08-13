# Vizuelno prepoznavanje govora pomoću LipNet modela

Studentski projekat iz predmeta Mašinsko učenje 2. Cilj praktičnog dela je pokretanje postojeće LipNet implementacije, analiza video pipeline-a, manji eksperimenti sa rezolucijom i augmentacijama i fino podešavanje modela na srpskom delu AI-SPEAK korpusa.

Trenutni status: projekat je vraćen na početak po VIPL-first roadmapu. Postojeći
Faza 1/Faza 2 kod, notebook-ovi i statični manifest/ROI artefakti su legacy i ne
koriste se kao osnova nove implementacije.

- [Novi VIPL-first roadmap](docs/analiza-i-roadmap.md)
- Legacy: [prethodno uputstvo za Fazu 1](docs/faza-1-colab.md)
- Legacy: [prethodno uputstvo za Fazu 2](docs/faza-2.md)
- [Tekst zadatka](36%20Vizuelno%20prepoznavanje%20govora%20na.txt)

Profesor je obezbedio `processed.zip` kao ulazni korpus za projekat. Arhiva, raspakovani `processed/` podaci i težine modela ostaju lokalno i namerno su izuzeti iz Git-a. Realizacija počinje od LipNet inference-a i narednih eksperimentalnih koraka; priprema ili preuzimanje korpusa nisu deo projekta.
