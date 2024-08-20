wget https://us.openslr.org/resources/12/train-clean-100.tar.gz
wget https://us.openslr.org/resources/12/dev-clean.tar.gz
wget https://us.openslr.org/resources/12/dev-other.tar.gz
wget https://us.openslr.org/resources/12/test-clean.tar.gz
wget https://us.openslr.org/resources/12/test-other.tar.gz

for file in *.tar.gz; do
    tar -zxvf $file
done

mv LibriSpeech datasets

rm *.tar.gz
