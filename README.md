In the following 3 models are tested against each other. All of them are deep reinforcement learning models trying to solve randomly created 8x8 Gridworld problems with random start and goal positions. All models see the agent’s current position and the 4 tiles surrounding them. The first model has no memory, therefore stumbling in the dark about where it already has been. The second model knows for the 4 surrounding tiles how often they have already been visited. The third model has an LSTM part integrated into its agent, allowing it to learn useful knowledge on its own. 

The files starting with GridworldA2CRandom contain the general training code as well as the specified hyperparameters. 
The files starting with GridworldAgentEnvRandom contain the agent and environment classes needed for running the general training file. They need not be run by the user.
The ending of the files show if the model either uses LSTM or handmade memory/no memory.
There is also a pdf-file in this repository containing first test results.
