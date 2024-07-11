###################################################################################README##################################################


There are 4 notebooks associated with Ms-Pacman game. 

1) buildAndTrainAgent-MsPacman : 

### 1) Importing Libraries and Packages: The Notebook starts by importing necessary libraries and packages such as gym, matplotlib, 
	   TensorFlow, and TF-Agents.

### 2) Setting up the Environment: It imports the MsPacman environment from OpenAI's Gym using 'gym.make('MsPacman-v4')
###    It defines sunctions to display obervations and randomly sample actions. 

### 3) Understanding the Environment: It explores the observation and action space of the MsPacman environment. 
### 								  It shows the initial observation and demonstartes taking a random action in the environment. 

### 4) Preprocessing the Environment: It preprocesses the environment using the TF-Agents, which includes frame stacking and action repeat. 

### 5) Creating the Q-Network: It defines a Q-Network using convolutional layers and a dense layer. 

### 6) Creating the DQN Agent: It creates a DQN agent using the Q-Network and specified parameters (eg. optimizer, target update period). 

### 7) Replay Buffer and Metrics: It sets up a replay buffer to store experiences and defines metrics for training. 

### 8) Training the Agent: It defines functions for training the agent and runs the training loop for a specified number of iterations. 

### 9) Visualizing Training: It creates animations to visulaize the agent's performance during training. 

### 10) Plotting Metrics: It plots the average return and average episode length over training iterations. 

Saving the Trained Model and Policy: 
It generates a video of the trained policy playing the MsPacman game. 

2) ResumeTraining: 
   Differences from Previous Code:

This code extends the functionality to include resuming training from a checkpoint and exporting/importing policies.
It introduces checkpointing, allowing the model to be saved and resumed from a specific point in training.
It saves the final trained model and policy.
								
3) DeployGamePlayer: 
   
This notebook mainly focuses on deploying a pre-trained agent and visualizing its performance. 
The environment, Q-Network, DQN Agent, and data collection mechanisms are already trained and saved in the pre-trained policy.
The policy_dir variable specifies the directory where the pre-trained policy is saved. In this case, it's a placeholder directory ("/var/tmp/policy").
The pre-trained policy is loaded using tf.saved_model.load(policy_dir).
The run_episodes_and_create_video() function takes the loaded policy, the evaluation environment, and the Python environment as inputs.
It runs the policy in the environment, captures frames, and creates a GIF to visualize the agent's performance.

4) Createtrainingcurvesandvideos: 

#Importing Libraries and Setting Up:

The necessary libraries are imported, and configurations are set up, including installing TensorFlow and configuring visualization settings.
Setting Hyperparameters and Environment:

Hyperparameters like learning rate, replay buffer capacity, and network layer sizes are defined. The MsPacman environment is loaded both for training and evaluation.
Defining the Q-Network and Agent:

The Q-network is defined using a neural network, and the DQN agent is created using this network, along with other specified parameters like optimizer and target update period.
Data Collection Setup:

Replay buffer and a driver for data collection are set up to collect experiences during training.
Training the Agent:

The agent is trained using the DQN algorithm, and a training function is defined to perform one training iteration.
Video Generation Function:

A function is defined to run episodes using a policy and generate a GIF video of the gameplay.
Checkpoint Handling:

A checkpointer is set up to save and restore agent checkpoints during training.
Training Curves:

Training curves (average return vs. cumulative gameplay experience) are plotted using pre-defined data.
Video Generation from Checkpoints:

Checkpoints from different stages of training are evaluated, and videos of gameplay using the trained policy are generated.
