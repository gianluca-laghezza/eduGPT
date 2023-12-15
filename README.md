# gpt

An educational implementation of the gpt model that follows Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

- gpt.py implements a decoder-only transformer model as described in [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- train.py is the training script where you can override the model hyper parameters and configs
- input.txt is the tiny Shakespeare dataset used as the training set

---

## Usage

You can simply run train.py in the console. If you want to change any config/hyper-parameter you can pass it as an argument to train.py as shown below

```console
$ python train.py --device='cuda' --compile=False --out_dir='16_layers' --n_layer=16 --log_interval=20
```

 
