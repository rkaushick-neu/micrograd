# Micrograd
Building neural networks from scratch.

This repository is built to replicate the existing [micrograd library](https://github.com/karpathy/micrograd) build by Andrej Karpathy.

## Value Object
The Value object is the smallest unit in a neural network. 

```mermaid
classDiagram
    class Value{
        + data: float
        + gradient: float
        + label: str
        # _prev: set[Value]
        # _operator: str
        + \_\_repr\_\_() str
        + \_\_add\_\_(other: Value) Value
        + \_\_mul\_\_(other: Value) Value
        + \_\_sub\_\_(other: Value) Value
        + \_\_truediv\_\_(other: Value) Value
    }
```

## The Neuron
WIP

## The Network Layer
WIP

## The Neural Network
WIP

## Forward Propagation 
WIP

## Back Propagation
WIP again.

## Neural Network Training
You guessed it, WIP.