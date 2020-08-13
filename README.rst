Installation notes
==================

This package was created and built with poetry_ and requires Python 3.7.
To use this package,

1. install poetry ( `--> instructions <https://python-poetry.org/docs/#installation/>`_)
2. clone this repository
3. change directory to your local repo with "cd path/to/repo"
4. enter "poetry install"

Step 4 will install all dependencies and automatically creates a virtual environment. (The command line
output provides the path to this virtual environment.) Activate the virtual environment with
"source path/to/virtualenv/bin/activate". To run unit tests, simply enter "pytest". To run the included
Jupyter Notebook, start a notebook server by entering "jupyter-notebook".


Bayesian Statistics
===================
The most commonly used definition of probability is the limit of a frequency. For example, if
you were to determine the probability for heads in a toin coss, you would toss the coin several
times and count how many times you see heads. For an infinite number of coin tosses, the relative
frequency of heads converges to the probability. However, using this definition, we would very soon
face severe limitations, as we could not answer simple but legitimate questions like this one:
how likely is it that there will be rain tomorrow?

There is no experiment that could be performed,
moreover, we cannot repeat it, as there is "only one tomorrow". Obviously, probability can also
be understood as a certain degree of believe. And this is were the Bayesian definition of probability
comes into play. It is built on Bayes theorem, which connects the conditional probability of
an event A given another event B, with the inverted conditional probability (p of event B, given
event A). This gives us the tools to estimate the probability of rain, as we can now use
historic data to do so. The probability of rain, given certain data (e.g. air pressure)
p(rain | data) can be expressed as the product of p(data | rain) (aka the likelihood) and a
general probability of a rainy day (aka our prior believe). This product is normalised by the
independent probability of observing the given data. In that, the Bayesian definition of
probability can be seen as un update of our current degree of believe with the help of current
(new) data.

While the frequentist approach to probability is taught more frequently (pun intended) and
usually computationally more efficient, frequentist approaches are prone to overfitting and as the
example above has shown, limited in scope.

Multi-armed bandit problem
==========================
This is an optimisation problem in which limited resources are supposed to be distributed to
several targets in such a fashion as to achieve maximal returns. The critical aspect is,
that initially it is not known which target is the most profitable, so there is a conflict between
**exploitation** of the currently chosen target and **exploration** of other targets.

This problem was coined the "multi-armed bandit problem" because it provides a colorful example: Imagine you
go to a casino and are willing to spend 100$ on playing slot machines. You suppose that the
different slot machines (aka one-armed bandits) have different success probabilities and so you kind
of want to test all of them - however, that means potentially wasting money on less successful machines ...

So what is the best tradeoff between exploration and exploitation? **Thompson sampling** provides
a **Bayesian** approach towards solving this problem.

Beta-Bernoulli model
====================
The Bernoulli distribution is a simple probability distribution: it provides the probability for
a random variable with only two possible outcomes, e.g. 0 or 1, on or off, heads or tails etc. A
slot machine is another example - you play and either you win, or you loose. Now suppose the chance
of winning, call it *mu*, is not known to you. You would need to play the slot machine several times
and keep track of your wins and losses.

The frequentist solution would be to simply estimate the success probability
from the relative frequency of wins. The Bayesian approach assumes a distribution for the parameter *mu* and
updates this distribution as new data are collected. This update involves the multiplication
of the prior distribution with the likelihood function (the Bernoulli distribution). It is handy, if the prior
distribution is chosen in a way so that the posterior has the same mathematical form as the prior.
This condition is fulfilled by the Beta distribution.

In summary, we end up using the **Beta-Bernoulli** model which is a Bernoulli model whose parameter *mu* is
distributed according to a beta distribution. The beta distribution has two parameters, a and b, so these
parameters are updated in the Bayesian procedure. Every time we observe a win, a is updated to a+1 and every time
we observe a loss, b is updated to b+1 (while the other parameter remains unchanged, respectively).

Thompson sampling
=================
Thompson sampling is an algorithm for solving the **Multi-armed Bandit Problem** which uses the **Beta-Bernoulli** model.
Remember, the goal is to find the best slot machine to play without loosing too much money on playing the "wrong" slot
machines. We assume that the success rates of all bandits are distributed according to a Beta distribution. Given an
initial set of the hyper-parameters a and b we **sample** a success rate for each slot machine and subsequently play
the slot machine with the highest success rate. We note whether we did win or loose in this particular game by updating
the parameters a and b for the currently selected slot machine.

In the next rounds we continue in the same fashion:
sample success rates, pick slot machine with highest success rate, update distribution of success rates. The most important
feature of this algorithm is the sampling of success rates. This allows us to, from time to time, play slot machines
that appear to be less successful according to our currently available data - so we keep on exploring in the initial
phase of our experiment. However, in time, all posterior distributions become more narrow (have a smaller variance) which
means that exploration becomes less and less likely and the algorithm converges to take actions on the more successful
slot machines.

.. _poetry: https://python-poetry.org/