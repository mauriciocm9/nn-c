## Neural Network in C

This is a toy project to teach me how to learn C.
Basically this is a replica of numpy but in plain C.
For now we have a linear regression model working for the MNIST dataset but eventually i want to create a basic neural network. Future tasks would include some CUDA kernels.

## Tests

N-cc is tested using [unity testing framework](https://github.com/ThrowTheSwitch/Unity)

```bash
git clone https://github.com/ThrowTheSwitch/Unity tests/unity
```
In order to get it working clone this repo into `tests/unity` folder.

After that you can execute
```bash
gcc tests/unity/src/unity.c tests/*.c -o tests -Itests/unity/src -o _tests; ./_tests
```