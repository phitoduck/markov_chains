# Markov Chains

The markov_chains.py file has several well 
documented functions and a class that generates 
fun sentences from the three text files.






This header is found in the file:

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

