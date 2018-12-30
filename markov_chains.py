# markov_chains.py

"""

Markov Chains

by Eric Riddoch
October 30, 2018

Description:

This file contains several functions that demonstrate
the creation and use of Markov matrices. At the end, of the file,
I use a Markov matrix to generate a few mixed lines of speech
in the style of Taylor Swift, Donald Trump, and Master Yoda.

File Contents:
    random_chain() - generates a random, valid markov chain
    forecast() - simulates calculating the weather (hot or cold) of
                    a set amount of days from a hard coded 2x2 markov chain
    four_state_forecast() - does the same as forecast() but with a 
                            4 state Markov Matrix
    steady_state() - given an input matrix, A, and a tolerance, this calculates
                        the matrix to which A converges
    
    SentenceGenerator (class) 
            Takes a path to a text file and creates a Markov
            matrix out of the lines of the file. Generates new
            lines based on the matrix with the babble method.


"""

import numpy as np
from colorama import Fore
import scipy.linalg as la


def random_chain(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. 
    
    @parameters:
        n (int): number of rows and columns of the resultant matrix

    @returns:
        an nxn 2-d numpy array whose columns sum to 1
    """
    
    # generate a random n × n matrix
    markov_matrix = np.random.random(size=(n, n))
    
    # make each column sum to 1 by dividing each column by column sum
    col_sums = markov_matrix.sum(axis=0)
    return markov_matrix / col_sums


def forecast(days):
    """Forcast the weather for the next 'days' days.
    
    Weather Matrix: P(Hot Tomorrow | Hot Today) = P(Row | Column)

                Hot Today ⇒ 
    Hot         [ .7   .6]
    Tomorrow?   [ .3   .4]
    
    @parameters:
        days (int): number of days to forcast

    @returns:
        List of 1's and 0's. Each index represents a day. A 1 in an index
        means it is hot, 0 if cold.

    """

    transition = np.array(
        [[0.7, 0.6], 
        [0.3, 0.4]]
    )

    # assume the first day was hot
    weather = list([0])

    # for each day, calculate tomorrow's weather based on today's weather
    for _ in range(days):
        result = np.random.binomial(1, transition[1, weather[-1]])
        weather.append(result)

    # we don't want the first day:
    weather.pop(0)

    # Sample from a binomial distribution to choose a new state.
    return weather


def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    The transition matrix

              Hot  Mild Cold Freezing
    Hot      [0.5, 0.3, 0.1, 0  ],
    Mild     [0.3, 0.3, 0.3, 0.3],
    Cold     [0.2, 0.3, 0.4, 0.5],
    Freezing [0,   0.1, 0.2, 0.2]

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]

    @parameters:
        days (int): number of days to forcast

    @returns:
        List of digits 0-3. A number in an index represents the weather on that day.
        0 - Freezing
        1 - Cold
        2 - Mild
        3 - Hot

    """
    
    transition = np.array(
        [
            [0.5, 0.3, 0.1, 0  ],
            [0.3, 0.3, 0.3, 0.3],
            [0.2, 0.3, 0.4, 0.5],
            [0,   0.1, 0.2, 0.2]
        ])
    
    # assume that the first day is mild
    weather = [1]

    for i in range(days):
        # get probabilities based on today's weather (column)
        probabilities = transition[:, weather[-1]]

        # calculate tomorrow's weather:
        # multinomial returns an index in the probabilites array based on what it lands on
        result = np.random.multinomial(1, probabilities)
        result = list(result).index(1)

        weather.append(result)

    return weather[1:]

def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    @parameters:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol       (float): The convergence tolerance.
        N           (int): The maximum number of iterations to compute.

    @raises:
        ValueError: if the iteration does not converge within N steps.

    @returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    
    m, n = A.shape

    # generate a state distribution vector x (entries sum to 1)
    x = np.random.random(size=(n))
    x /= x.sum() 

    # matrix vector multiply N times
    for i in range(N):
        Ax = A @ x

        # return product is within the tolerance of x
        if la.norm(x - Ax) < tol:
            return Ax
        else:
            x = Ax

    raise ValueError("The given matrix does not converge!")

        

class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """
    
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        
        # read all lines
        with open(filename, encoding="utf8") as infile:
            self.sentences = infile.readlines()

        # split each line into words
        for i in range(len(self.sentences)):
            self.sentences[i] = ['$tart'] + self.sentences[i].rstrip('\n').split() + ['$top']

        # find unique words
        unique_words = []
        for sentence in self.sentences:
            for word in sentence:
                if word not in unique_words:
                    unique_words.append(word)

        # initialize transition matrix (n + 2 for $tart-$top)
        n = len(unique_words) - 2 # - 2 to account for Start and Stop already being in set
        transition = np.zeros(shape=(n + 2, n + 2))
        
        # initialize dictionary of indices
        states = []
        states.append("$tart")
        for word in unique_words:
            if word != "$top" and word != "$tart":
                states.append(word)
        states.append("$top")

        indices = dict()
        for i in range(len(states)):
            indices[states[i]] = i

        # add 1 to start and stop states
        transition[n + 1, n + 1] = 1 # $top transitions to self

        self.dummy = transition.copy()

        # build transition matrix
        for sentence in self.sentences:
            for i in range(len(sentence)):
                if i < (len(sentence) - 1):
                    col_index = indices[sentence[i]]
                    row_index = indices[sentence[i + 1]]
                    transition[row_index, col_index] += 1

        self.no_probs = transition.copy()
        self.words = unique_words

        # make each column sum to 1
        transition /= transition.sum(axis=0)

        self.transition = transition
        self.indices = indices
        self.states = states

    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """

        words = []
        current_word = "$tart"

        while current_word != "$top":
            # get probabilities based on current word (column)
            probabilities = self.transition[ :, self.indices[current_word] ]

            # calculate next word:
            # multinomial returns an index in the probabilites array based on what it lands on
            trial = np.random.multinomial(1, probabilities)
            index_of_next_word = list(trial).index(1) # locates index of succesful trial
            current_word = self.states[index_of_next_word]

            # add the current_word
            if current_word not in ["$tart", "$top"]:
                words.append(current_word)

        return " ".join(words)
