#!/bin/bash
wget -O Downloaded_files.zip "https://www.dropbox.com/scl/fo/tqdf47fae9ym7tt0bskzs/AMANmKs2TKtUPDtxpageByw?rlkey=ke5kwzirpktr3djm9j24ychn8&st=ojw1bexh&dl=1"
wget -O Efficiency_files.zip "https://www.dropbox.com/scl/fo/plemkwh67symbpaosmwel/AMotU2eYm1IZEkacQaF7s4M?rlkey=0ustfs0j6h6op2ip8ku6ds3ok&st=gaimp0df&dl=1"
wget -O Preprocessed_files.zip "https://www.dropbox.com/scl/fo/gb0va4iayuio4k11icw4g/AJnLj70nDuzBkcgUVNdlLtM?rlkey=a95gemfgm1fyrcbzfeis97gq4&st=ulywk0f9&dl=1"
unzip Downloaded_files.zip -d Downloaded_files/
unzip Efficiency_files.zip -d Efficiency_files/
unzip Preprocessed_files.zip -d Preprocessed_files/
mkdir -p data/
mv Downloaded_files/ Efficiency_files/ Preprocessed_files/ data/
rm Downloaded_files.zip Efficiency_files.zip Preprocessed_files.zip
