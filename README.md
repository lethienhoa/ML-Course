### Lecturer

Hoa T. Le

Contact me at <first_name>.<last_name>@loria.fr 
or at my offfice B213 (Loria) (please make an appointment first).

# Overview

The aim of this course is to introduce computational, numerical and distributed memories from a theoretical and epistemological standpoint as well as neural networks and their use in cognitive science. Concerning machine learning, the course will focus on various model learners such as Markov Chains, Hidden Markov Model, Reinforcement Learning and Neural Networks.

# Target audience

This course is for Master 1 Science Cognitive and Applications (University of Lorraine). This is an introduction course, assuming no prior knowledge of Machine Learning.

<p>
  <img src="https://github.com/lethienhoa/Memory-and-Machine-Learning-Course/blob/master/logo_ul.png?raw=true" />
</p>

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
- Numerical Optimization. Jorge Nocedal and Stephen J. Wright (1999). 
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

__Reading__
* [Chapters 5, 7 of Deep Learning Book.](http://www.deeplearningbook.org/)

__`Practical`: Training an Image Classifier on CIFAR10 data from scratch [(TP 1)](https://docs.google.com/presentation/d/1fpqi7tPWUft8N1wmoMrKiTUbZsf6ccOsiiBZAGqT4ps/edit?usp=sharing)__
* Define the network
* Loss function
* Backprop
* Update the weights

__`Prerequisite`: Linear Algebra__
* [Chapter 2 of Deep Learning Book.](http://www.deeplearningbook.org/)
* Linear algebra book [(a good book on this subject)](https://www.google.fr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=7&cad=rja&uact=8&ved=0ahUKEwji5YO7warZAhUK7RQKHaEnCxMQFghfMAY&url=https%3A%2F%2Fwww.math.ucdavis.edu%2F~linear%2Flinear-guest.pdf&usg=AOvVaw17vykn2bAuZTvQzDMstEzg)

## Lecture 3-4. Optimization [(Slides)](https://drive.google.com/open?id=19H3UWwfXtVJ3WYVNfI6IcNyEvXTTpEXAAjZL45Bw1fA) [(Revision)](https://drive.google.com/open?id=1zSe6MRdxr7my60VMshAbihuQ98v3e2GwlWMA8CWbQwI)

* Linear Least Squares Optimization
  * Cholesky decomposition
  * QR decomposition
* Iterative methods  
  * Steepest gradient descent
  * Momentum, Nesterov
  * Adaptive learning rates (Adagrad, Adadelta, Adam)

__Reading__
* [Chapters 4, 8 of Deep Learning Book.](http://www.deeplearningbook.org/)

__`Practical`: Neural Networks for Text [(TP 2)](https://docs.google.com/presentation/d/1eLQlGjwJW7a7s818m8Q9OrUB-KejtI1zL_pg2wRyNX8/edit?usp=sharing)__
* Text Classification with Logistic Regression on BOW Sentence representation
* Text Classification with Word Embeddings
* N-Gram Language Modeling and Continuous BOW

__`Prerequisite`:__
* Numerical optimization book (Nocedal and Wright)
* [Bag-of-words Sentence Representation](https://en.wikipedia.org/wiki/Bag-of-words_model)
* [Word Embeddings](https://machinelearningmastery.com/what-are-word-embeddings/)


## Lecture 5. Neural Network [(Slides)](https://docs.google.com/presentation/d/19BWO7yvnSGUHFBAd8BYzZstB1TxLkGBBY2wemhtPby8/edit?usp=sharing)

* Feed Forward Neural Network
* Backpropagation
* Recurrent Neural Network

__Reading__
* [Chapters 6, 10 of Deep Learning Book.](http://www.deeplearningbook.org/)

__`Practical`: [(TP 3)](https://drive.google.com/open?id=1ss05RmBRk5etcbhqkLzh58bGlZqxnUUqCNJSTa6EfCo)__
* [RNN for Part-of-Speech Tagging](https://drive.google.com/open?id=1qZnCS3T2wOAxCfEldA9MlHvPH_NoTT51)
* [RNN for Language Modeling (optional)](https://github.com/pytorch/examples/tree/master/word_language_model)

__More Reading__
* [Deep Neural Networks for Acoustic Modeling in Speech Recognition. IEEE Signal Processing Magazine 2012](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/HintonDengYuEtAl-SPM2012.pdf)
* [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation. Arxiv 2016](https://arxiv.org/pdf/1609.08144.pdf)


## Lecture 6. Long-Short Term Memory Networks [(Slides)](https://drive.google.com/open?id=14EJEeWiGIwM9jPsmbIRhqnvfI7gQLCYS2Ob9OTCyfnY)

* Vanishing gradient problem of RNN
* Training recurrent networks (activation functions, gradient clipping, initialization,...)
* LSTM (Stacked LSTMs, BiLSTM)
* Sequence-to-Sequence model for Machine Translation

__Reading__
* [Chapter 10 of Deep Learning Book.](http://www.deeplearningbook.org/)

__`Practical`: [(TP 4)](https://docs.google.com/presentation/d/1hIvVu__GAMJIyC0W8kS_5HlnfCgYsge5XCIjzp2N-xg/edit?usp=sharing)__
* Translation with a Sequence to Sequence Network and Attention (from scratch)

__More Reading__
* [Google Neural Machine Translation tutorial (in tensorflow)](https://github.com/tensorflow/nmt)


## Lecture 7. Autoencoders

* Autoencoder (AE)
* Denoising Autoencoder (DAE)
* Variational Autoencoder (VAE)
* Adversarial Autoencoder (AAE)

__Reading__
* [Chapter 14 of Deep Learning Book.](http://www.deeplearningbook.org/)

__`Practical`:__
* OpenNMT PyTorch 1 (Data Loaders)
* OpenNMT PyTorch 2 (Framework and Modules)

__More Reading__
* **(CNN-DCNN) Autoencoder (AE)**: Yizhe Zhang, Dinghan Shen, Guoyin Wang, Zhe Gan, Ricardo Henao, Lawrence Carin. Deconvolutional Paragraph Representation Learning. NIPS 2017
* **(Sequential) Denoising Autoencoder (DAE)**: Felix Hill, Kyunghyun Cho, Anna Korhonen. Learning Distributed Representations of Sentences from Unlabelled Data. NAACL-HLT 2016
* **Variational Autoencoder (VAE)**: Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz, Samy Bengio. Generating Sentences from a Continuous Space. CoNLL 2016
* **Adversarial Autoencoder (AAE)**: Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow. Adversarial Autoencoders. ICLR 2016
