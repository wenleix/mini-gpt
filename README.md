# mini_gpt

## Setup pyenv
The following commands only need to run once on the laptop.
```
# Make sure running in arm64 mode
arch

brew install pyenv
brew install pyenv-virtualenv
pyenv install 3.11.4
```

To clone and setup pyenv for this repo:
```
# Clone repo
git clone git@github.com:wenleix/mini_gpt.git
cd mini_gpt

# Setup pyenv
pyenv virtualenv 3.11.4 mini_gpt-3.11.4
pyenv local mini_gpt-3.11.4

# Install requirements
pip install -r requirements.txt
```

## Misc
```
python train.py config/train_shakespeare_char.py \
	--device=mps \
	--compile=False \
	--eval_iters=20 \
	--log_interval=1 \
	--block_size=256 \
	--batch_size=64 \
	--n_layer=4 \
	--n_head=4 \
	--n_embd=128 \
	--max_iters=2000 \
	--lr_decay_iters=2000 \
	--dropout=0.0 \
	--weight_decay=0.0
```

