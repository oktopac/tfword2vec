# tfword2vec
A fork of the tensor flow word2vec example

# Examples

```
pushd /tmp && curl http://mattmahoney.net/dc/enwik8.zip -o enwik8.zip && unzip enwik8.zip && head -n 100000 enwik8 > enwik8.100000 && popd
git clone https://github.com/oktopac/tfword2vec.git
cd tfword2vec
virtualenv env -p python3
python setup.py install
python examples/wiki8.py /tmp/wiki8.100000 /tmp/output/ --epochs=10000
tensorboard --logdir=/tmp/output
```
