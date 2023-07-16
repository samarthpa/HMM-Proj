import numpy as np
import operator

def part_1_a():
    """Provide probabilities for the word HMMs outlined below.
    Word ALLIGATOR, NUTS, and SLEEP.
    Returns:
        tuple() of
        (prior probabilities for all states for word ALLIGATOR,
         transition probabilities between states for word ALLIGATOR,
         emission parameters tuple(mean, std) for all states for word ALLIGATOR,
         prior probabilities for all states for word NUTS,
         transition probabilities between states for word NUTS,
         emission parameters tuple(mean, std) for all states for word NUTS,
         prior probabilities for all states for word SLEEP,
         transition probabilities between states for word SLEEP,
         emission parameters tuple(mean, std) for all states for word SLEEP)
        Sample Format (not complete):
        (
            {'A1': prob_of_starting_in_A1, 'A2': prob_of_starting_in_A2, ...},
            {'A1': {'A1': prob_of_transition_from_A1_to_A1,
                    'A2': prob_of_transition_from_A1_to_A2,
                    'A3': prob_of_transition_from_A1_to_A3,
                    'Aend': prob_of_transition_from_A1_to_Aend},
             'A2': {...}, ...},
            {'A1': tuple(mean_of_A1, standard_deviation_of_A1),
             'A2': tuple(mean_of_A2, standard_deviation_of_A2), ...},
            {'N1': prob_of_starting_in_N1, 'N2': prob_of_starting_in_N2, ...},
            {'N1': {'N1': prob_of_transition_from_N1_to_N1,
                    'N2': prob_of_transition_from_N1_to_N2,
                    'N3': prob_of_transition_from_N1_to_N3,
                    'Nend': prob_of_transition_from_N1_to_Nend},
             'N2': {...}, ...}
            {'N1': tuple(mean_of_N1, standard_deviation_of_N1),
             'N2': tuple(mean_of_N2, standard_deviation_of_N2), ...},
            {'S1': prob_of_starting_in_S1, 'S2': prob_of_starting_in_S2, ...},
            {'S1': {'S1': prob_of_transition_from_S1_to_S1,
                    'S2': prob_of_transition_from_S1_to_S2,
                    'S3': prob_of_transition_from_S1_to_S3,
                    'Send': prob_of_transition_from_S1_to_Send},
             'S2': {...}, ...}
            {'S1': tuple(mean_of_S1, standard_deviation_of_S1),
             'S2': tuple(mean_of_S2, standard_deviation_of_S2), ...}
        )
    """
    """Word ALLIGATOR"""
    a_prior_probs = {
        'A1': 0.333,
        'A2': 0.,
        'A3': 0.,
        'Aend': 0.
    }
    a_transition_probs = {
        'A1': {'A1': 0.833, 'A3': 0., 'A2': 0.167, 'Aend': 0.},
        'A2': {'A1': 0., 'A2': 0.786, 'A3': 0.214, 'Aend': 0.},
        'A3': {'A2': 0., 'A3': 0.727, 'A1': 0., 'Aend': 0.273},
        'Aend': {'A1': 0., 'A3': 0., 'A2': 0., 'Aend': 1}
    }
    # Parameters for end state is not required͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏󠄇͏︀
    a_emission_paras = {
        'A1': (51.056, 21.986),
        'A2': (28.357, 14.936),
        'A3': (53.727, 16.707),
        'Aend': (None, None)
    }

    """Word NUTS"""
    n_prior_probs = {
        'N1': 0.333,
        'N2': 0.,
        'N3': 0.,
        'Nend': 0.
    }
    # Probability of a state changing to another state.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏󠄇͏︀
    n_transition_probs = {
        'N1': {'N3': 0., 'N1': 0.919, 'N2': 0.081, 'Nend': 0.},
        'N2': {'N3': 1, 'N1': 0., 'N2': 0., 'Nend': 0.},
        'N3': {'N3': 0.625, 'N1': 0., 'N2': 0., 'Nend': 0.375},
        'Nend': {'N3': 0., 'N2': 0., 'N1': 0., 'Nend': 1}
    }
    # Parameters for end state is not required͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏󠄇͏︀
    n_emission_paras = {
        'N1': (38.081, 11.175),
        'N2': (42, 2.828),
        'N3': (60, 13.491),
        'Nend': (None, None)
    }

    """Word SLEEP"""
    s_prior_probs = {
        'S1': 0.333,
        'S2': 0.,
        'S3': 0.,
        'Send': 0.
    }
    s_transition_probs = {
        'S1': {'S2': 0.375, 'S3': 0., 'S1': 0.625, 'Send': 0.},
        'S2': {'S1': 0., 'S2': 0.864, 'S3': 0.136, 'Send': 0.},
        'S3': {'S2': 0., 'S1': 0., 'S3': 0., 'Send': 1},
        'Send': {'S2': 0., 'S3': 0., 'S1': 0., 'Send': 1}
    }
    # Parameters for end state is not required͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏󠄇͏︀
    s_emission_paras = {
        'S1': (29.5, 8.411),
        'S2': (36.182, 5.990),
        'S3': (36.667, 1.886),
        'Send': (None, None)
    }

    return (a_prior_probs, a_transition_probs, a_emission_paras,
            n_prior_probs, n_transition_probs, n_emission_paras,
            s_prior_probs, s_transition_probs, s_emission_paras)

def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    """
    if list(para_tuple) == [None, None]:
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile



def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.
    Args:
        evidence_vector (list): List of right hand Y-axis positions (integer).
        states (list): List of all states in a word. No transition between words.
                       example: ['A1', 'A2', 'A3', 'Aend', 'N1', 'N2', 'N3', 'Nend']
        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}
        transition_probs (dict): dictionary representing transitions from each
                                 state to every other valid state such as for the above
                                 states, there won't be a transition from 'A1' to 'N1'
        emission_paras (dict): parameters of Gaussian distribution
                                from each state.
    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )
    """
    sequence = []
    probability = 0.0

    #check if eveidence vector is empty
    if len(evidence_vector) == 0:
        return sequence, probability

    #initialize variables with np zeros, KxT tables
    T1 = np.zeros((len(states), len(evidence_vector)))
    T2 = np.zeros((len(states), len(evidence_vector)), dtype=int)

    #using DP approach, note [i, j] are reversed compared to psedocode in folder
    for i in range(len(evidence_vector)):
        for j in range(len(states)):
            for k in range(len(states)):
                T1[k, 0] = prior_probs[states[k]] * gaussian_prob(evidence_vector[0], emission_paras[states[k]])
                if states[j] in transition_probs[states[k]]:
                    prob = T1[k, i-1] * transition_probs[states[k]][states[j]] * gaussian_prob(evidence_vector[i], emission_paras[states[j]])
                if prob >= T1[j, i]:
                    T1[j, i] = prob
                    T2[j, i] = k
    #as seen in pseudocode "The table entries are filled by increasing order"
    probability = np.max(T1[:, len(evidence_vector)-1])
    maxstate = np.argmax(T1[:, len(evidence_vector)-1])

    #loop to add state to list
    for i in range(len(evidence_vector)-1, -1, -1):
        sequence.append(states[maxstate])
        maxstate = T2[maxstate, i]

    sequence.reverse()

    return sequence, probability