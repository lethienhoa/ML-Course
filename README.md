### Lecturer

Hoa T. Le

Contact me at <first_name>.<last_name>@loria.fr 
or at my offfice B213 (Loria) (please make an appointment first).

# Overview

The aim of this course is to introduce computational, numerical and distributed memories from a theoretical and epistemological standpoint as well as neural networks and their use in cognitive science. Concerning machine learning, the course will focus on various model learners such as Markov Chains, Hidden Markov Model, Reinforcement Learning and Neural Networks.

# Target audience

This course is for Master 1 Science Cognitive and Applications (Master Erasmus Mundus – University of Lorraine). This is an introduction course, assuming no prior knowledge of Machine Learning.

# Course Organization

- 30 hours = 10 work sessions of 3 hours/week
- Courses = half lectures / half exercises or practicals
- Evaluation: individual project
- The last 2 (maybe 3) work sessions will be saved to work on the project

## 20% projects
You can choose one of these books, read (entirely or at least 5 chapters) and write a resume in one page.

__John Tabak´s series:__
- Probability and Statistics: The Science of Uncertainty (History of Mathematics)
- Algebra: Sets, Symbols, and the Language of Thought (History of Mathematics) 
- Geometry: The Language of Space and Form (History of Mathematics)
- Beyond Geometry: A New Mathematics of Space and Form (History of Mathematics)
- Numbers: Computers, Philosophers, and the Search for Meaning (History of Mathematics)
- Mathematics and the Laws of Nature: Developing the Language of Science (History of Mathematics)

__Michael Guillen:__
- Five Equations That Changed the World: The Power and Poetry of Mathematics

__Ian Stewart:__
- In Pursuit of the Unknown: 17 Equations That Changed the World

## Book References
- Reinforcement Learning: An Introduction. Richard S. Sutton and Andrew G. Barto (1998). 
- Iterative Methods for Optimization. C. T. Kelley (1999). 
- The Elements of Statistical Learning. H. Friedman, Robert Tibshirani and Trevor Hastie (2001). 
- Inference in Hidden Markov Models. Olivier Cappé, Eric Moulines and Tobias Rydén (2005). 
- Pattern Recognition and Machine Learning. Christopher M. Bishop (2006). 
- Deep Learning. Ian Goodfellow, Yoshua Bengio and Aaron Courville (2016). 

# Syllabus

## Lecture 1. Introduction about Artificial Intelligence [(Slides)](https://docs.google.com/presentation/d/1QXT02QAzS3hwYMW32NtI-AAzkYzuc8FBBxK5Sg1UPqY/edit?usp=sharing)

__Reading__
* [Deep learning. Yann LeCun,	Yoshua Bengio	& Geoffrey Hinton. Nature 2015.](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)
* [Human-level control through Deep Reinforcement Learning. Mnih et al., Nature 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Mastering the game of Go with deep neural networks and tree search. Silver et al., Nature 2016.](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
* [Hybrid computing using a neural network with dynamic external memory. Graves et al., Nature 2016.](https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)
* [Neuroscience-Inspired Artificial Intelligence. Hassabis et al., Neuron 2017.](https://deepmind.com/documents/113/Neuron.pdf)

__More Reading__
* [The Rise of Computer-Aided Explanation. Nielsen. QuantaMagazine 2015.](https://www.quantamagazine.org/the-rise-of-computer-aided-explanation-20150723)
* [Will Computers Redefine the Roots of Math ? Hartnett. QuantaMagazine 2015.](https://www.quantamagazine.org/univalent-foundations-redefines-mathematics-20150519)
* [Mapping the Brain to Build Better Machines. Singer, QuantaMagazine 2016.](https://www.quantamagazine.org/mapping-the-brain-to-build-better-machines-20160406)

__`Practical`: Learning basic PyTorch [(open tutorial)](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)__
* What is PyTorch ?
* Initialization and matrix computation
* Conversion between PyTorch <-> Numpy
* Autograd: automatic differentiation package

__Installation instructions:__
* [How to Install Ubuntu 16.10/16.04 Alongside With Windows 10 or 8 in Dual-Boot](https://www.tecmint.com/install-ubuntu-16-04-alongside-with-windows-10-or-8-in-dual-boot/)
* [Install conda environment (if it is not yet installed)](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)
* [Install PyTorch](http://pytorch.org/)
* If Linux and PyTorch is properly installed, to run code, just open Linux terminal and call __'jupyter notebook'__

## Lecture 2. Baseline models and Loss functions [(Slides)](https://docs.google.com/presentation/d/1bNuD1P5ZAJGGwjfvsyUXEKNvb47hn9I2OkbZKXB4NFg/edit?usp=sharing)

* A classification’s pipeline
* K-Nearest Neighbors (KNN) 
* Linear Classifier
* Loss function
* Regularization

__`Practical`: Training an Image Classifier on CIFAR10 data from scratch [(TP 1)](https://docs.google.com/presentation/d/1fpqi7tPWUft8N1wmoMrKiTUbZsf6ccOsiiBZAGqT4ps/edit?usp=sharing)__
* Define the network
* Loss function
* Backprop
* Update the weights

__`Prerequisite`: Linear Algebra__
* Linear algebra cheat sheet for deep learning [(blog)](https://towardsdatascience.com/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c)
* Linear algebra for neural networks [(short and rigorous overview)](https://www.google.fr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=10&ved=0ahUKEwiLjIHNwarZAhXCSBQKHasPC64QFgiRATAJ&url=https%3A%2F%2Fwww.utdallas.edu%2F~herve%2FAbdi-lann-01.pdf&usg=AOvVaw1wg61a36SkaR4cQLrLmXAQ)
* Linear algebra book [(a good book on this subject)](https://www.google.fr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=7&cad=rja&uact=8&ved=0ahUKEwji5YO7warZAhUK7RQKHaEnCxMQFghfMAY&url=https%3A%2F%2Fwww.math.ucdavis.edu%2F~linear%2Flinear-guest.pdf&usg=AOvVaw17vykn2bAuZTvQzDMstEzg)
