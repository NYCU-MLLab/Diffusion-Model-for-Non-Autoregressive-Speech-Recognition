for file in *.tar.gz; do
    tar -zxvf $file
done

mv cv-corpus-17.0-2024-03-15/zh-TW datasets

rm -r cv-corpus-17.0-2024-03-15
rm *.tar.gz
