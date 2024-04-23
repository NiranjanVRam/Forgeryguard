#!/bin/bash

#EDIT these 3 paths:
#path to columbia database
#download: http://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/
COLUMBIA="C:/Users/HP/Downloads/Telegram Desktop/forensic-graph-master/datasets/columbia_uncompressed_database/"

#path to carvalho database
#download: http://ic.unicamp.br/~rocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip
CARVALHO="C:/Users/HP/Downloads/Telegram Desktop/forensic-graph-master/datasets/carvalho-tifs/carvalho-tifs-database/"

#path to korus database
#download: http://pkorus.pl/downloads/dataset-realistic-tampering
KORUS="/tampering-databases/korus-realistic-tampering-database/"


#make parent folder
mkdir -p ./tampering-databases

#make image and masks folders
mkdir -p ./tampering-databases/4cam_auth
mkdir -p ./tampering-databases/4cam_splc
mkdir -p ./tampering-databases/4cam_masks

mkdir -p ./tampering-databases/carvalho_pristine
mkdir -p ./tampering-databases/carvalho_tampered
mkdir -p ./tampering-databases/carvalho_masks

#mkdir -p ./tampering-databases/korus_pristine
#mkdir -p ./tampering-databases/korus_tampered
#mkdir -p ./tampering-databases/korus_masks

#Copy columbia files
echo "Copying columbia database"
cp "$COLUMBIA"4cam_splc/*.tif ./tampering-databases/4cam_splc/
cp "$COLUMBIA"4cam_auth/*.tif ./tampering-databases/4cam_auth/
cp "$COLUMBIA"4cam_splc/edgemask/*.jpg ./tampering-databases/4cam_masks/

#Copy Carvalho files
echo "Copying carvalho database"
cp "$CARVALHO"DSO-1/normal*.png ./tampering-databases/carvalho_pristine/
cp "$CARVALHO"DSO-1/splicing*.png ./tampering-databases/carvalho_tampered/
cp "$CARVALHO"DSO-1-Fake-Images-Masks/splicing*.png ./tampering-databases/carvalho_masks/

#Convert carvalho files to .tif
echo "Converting Carvalho originals to .TIF"
for i in ./tampering-databases/carvalho_pristine/*.png ; do convert "$i" "${i%.*}.TIF" ; done
echo "Converting Carvalho tampered to .TIF"
for i in ./tampering-databases/carvalho_tampered/*.png ; do convert "$i" "${i%.*}.TIF" ; done

#Copy Korus files
echo "Copying Korus database"
cp "$KORUS"pristine/*.TIF ./tampering-databases/korus_pristine/
cp "$KORUS"tampered/*.TIF ./tampering-databases/korus_tampered/
cp "$KORUS"masks/*.PNG ./tampering-databases/korus_masks/