# Diffusion Model for Non-Autoregressive Speech Recognition

## Environment

```
pip3 install -r requirements.txt
```

## Training and Inference

### Select Language

* English

  ```
  cd English
  ```

* Chinese

  ```
  cd Chinese
  ```

### Select Acoustic Encoder

* wav2vec 2.0

  ```
  cd wav2vec2
  ```

* HuBERT

  ```
  cd hubert
  ```

* WavLM

  ```
  cd wavlm
  ```

### Download Dataset

* English

  ```
  bash dataset.sh
  ```

* Chinese

  Download dataset from **[Common Voice](https://commonvoice.mozilla.org/en/datasets)** by selecting **Chinese (Taiwan)** and **Common Voice Corpus 17.0**

  ```
  bash dataset.sh
  ```

### Extract Feature

```
python3 extract.py
```

### Training

```
python3 train.py
```

### Inference

* Without fast sampling

  ```
  python3 inference.py
  ```

* With fast sampling

  ```
  python3 inference_2.py --C 0.99 --T 5
  ```
