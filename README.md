# eduGPT

An educational implementation of the gpt model that follows Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

- gpt.py implements a decoder-only transformer model as described in [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- train.py is the training script where you can override the model hyper parameters and configs
- input.txt is the tiny Shakespeare dataset used as the training set
- sample.py generates new text from the model

  When trained on the tiny Shakespeare dataset it achieves a log-loss of 2.0635. The hyperparameterTuning notebook contains some analysis on the hyperparameters used.

---

## Usage

You can simply run train.py in the console. If you want to change any config/hyper-parameter you can pass it as an argument to train.py as shown below

```console
$ python train.py --device='cuda' --compile=False --out_dir='16_layers' --n_layer=16 --log_interval=20
```

To generate text run sample.py as below
```console
$ python sample.py --device='cpu'
```
Some sample text generated (LOL):


heng we fithath spere.

CABENAMNCENA:
I of noire, low theng peled's as to-arden you plians of und
Buse oparstiong unt wadit Rusherory don.

KING HARD IIII:
The is whall hee?
Now and my heave decell an of sort.

SICKING EWARD IIV:
Buse me I clow on the I not hey, folear wit the so fold I don He the so allvan,
Give ang forme 'Tof that ou kin the with pay him arme on me oll.


NORORD:
The not pomaterulss an
Ty ou hat is I for gon this comere gat prour
O mom ding this co'd the which fight you off a

 
