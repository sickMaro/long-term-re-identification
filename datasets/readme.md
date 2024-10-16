aggiungere all'interno della directory le cartelle relative ai datasets che si intedono usare per il progetto nella forma:

nome_dataset
|
|-query
|-gallery

qualora il dataset abbia campioni suddivisi in molteplici giorni ci si aspetta una ripartizione del tipo:

nome_dataset
|
|-query
|-|
  |-query_day_one
  |-query_day_two
  |- ...
|-gallery
 |
 |-gallery_day_one
 |-gallery_day_two


- **cartella-principale/**: Cartella principale del progetto, contiene tutte le sottocartelle.
  - **sottocartella1/**: Contiene file di esempio.
  - **sottocartella2/**: Contiene altri file di configurazione.
- **documentazione/**: Cartella dedicata alla documentazione del progetto.
  - **guida.md**: File con la guida all'utilizzo del progetto.
