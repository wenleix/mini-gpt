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
