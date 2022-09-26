# Optimisation-Algorithms
A collection of the most commonly used Optimisation Algorithms for Data Science &amp; Machine Learning

---

This repository is created by and belongs to: https://github.com/Muradmustafayev-03

Contributing guide: https://github.com/Muradmustafayev-03/Optimisation-Algorithms/blob/main/CONTRIBUTING.md

To report any issues: https://github.com/Muradmustafayev-03/Optimisation-Algorithms/issues

To install the package as a library use: ***pip install optimisation-algorithms==1.0.0***

---

In this project I try to collect as many useful Optimisation Algorithms as possible, and write them in a simple and reusable way.
The idea is write all these algorithms in python, in a fundamental, yet easy-to-use way, with *numpy* being the only external library used.
The project is currently on the early stages of the development, but one can already try and implement it in their own projects.
And for sure, you are always welcome to contribute or to make any suggestions. Any feedback is appreciated.

## What is an Optimisation Algorithm?
*For more information: https://en.wikipedia.org/wiki/Mathematical_optimization*

**Optimisation Algorithm** is an algorithm, that is used to find input values of the *global minima*, or less often, of the *global maxima* of a function.

In this project, all the algorithms look for the *global minima* of the given function. 
However, if you want to find the *global maxima* of a function, you can pass the negation of your function: *-f(x)*, instead of *f(x)*, 
having that its mimima will be the maxima of your function.

**Optimization algorithms** are widely used in **Machine Learning**, **Mathematics** and a range of other applied sciences.

There are multiple kinds of **Optimization algorithms**, so here is a short description ofthe ones, used in the project:

### Iterative Algorithms
Tese algorithms start from a random or specified point, and step by step move torwards the closest minima. 
They often require the *partial derivative* or the *gradient* of the function, which requires function to be differentiable.
These algorithms are simple and work good for *bowl-like* functions, 
thow if there is mor than one minimum in function, it can stock at a *local minima*, insted of finding the *global* one.

##### Examples of the *Iterative Algorithms* used in this project are:
- Gradient Descent
  - Batch Gradient Descent
  - Approximated Batch Gradient Descent

### Metaheuristic Algorithms
These algorithms start with a set of random solutions, 
then competetively chose the best solutions from the set and based on them, 
generate a new set of better solutions, thus evolving each iteration.
These algorithms don't stock at *local minimums*, but directly find the *global* one, so they can be used for *many-local-minimums* functions.

##### Examples of the *Metaheuristic Algorithms* used in this project are:
- Harmony Search Algorithm
- Genetic Algorithm

## Benchmark Functions
*Benchmark functions* are used to test *Optimization Algorithms*, however they can be used by themselves. 
There are multiple *benchmark functions* used in this project, and they are divided into several types depending on their shape.

*For more information and the mathematical definition of the functions see: https://www.sfu.ca/~ssurjano/optimization.html*
