# Hidden-Markov-Models-Project

In this project, I built a word recognizer for American Sign Language (ASL) video sequences to demonstrate the power of probabilistic models. In particular, this project employs hidden Markov models (HMM's) to analyze a series of measurements taken from videos of isolated American Sign Language (ASL) signs collected for research.

In each video, an ASL signer signs a meaningful sentence. In a typical ASL recognition system, you observe the XY coordinates of the speaker's left hand, right hand, and nose for every frame. The following diagram shows how the positions of the left hand (Red), right hand (Blue), and nose (Green) change over time. Saturation of colors represents time elapsed.![hands_nose_position](https://github.gatech.edu/storage/user/64855/files/ab9a990a-2d54-4a61-806b-a30e34ad6adc)

In this specific project I built a model that is able to recognize the hand gestures of ASL and output the desired word.
