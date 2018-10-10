## CIFAR-10

_tags: visual recognition, tensorflow_

Based on: http://cs231n.github.io/assignments2018/assignment2/

The input data files can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html

- The codes are based on the assignments for the computer vision course offered at Stanford ([CS231n](http://cs231n.stanford.edu/syllabus.html)).

- Divided the code into _Model_ object and main codes to improve readability

- Tensorflow graph is built "manually", which allows for a more flexible neural network design

- The neural network was inspired by GoogLeNet design

- If you use keras-Sequential model, it's difficult to build complicated structures like GoogLeNet or ResNet (it's technically possible, but this way is more intuitive).


