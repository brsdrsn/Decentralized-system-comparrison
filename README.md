# Decentralized-system-comparrison
Comparison between a centralized and decentralized control system. 

Functions of each file:
Get_states.py: get the current state of the agent.
RNN_m2.py: Define the structure of the recurrent network used in this research.
C_Brain_Rnn.py: The centralized control system.
DC_Brain_rnn.py: The decentralized control system.
Optimizer_rnn.py: Used to be the main training file, now only used to initiate the simulation for the training files.(misleading name)
spiderbot_rnn.py: The main optimized for the centralized control system. Contains all the code related to the evolutionary algorithm.
spiderbot_decent_rnn.py: The main optimized for the decentralized control system. Contains all the code related to the evolutionary algorithm.
(the spiderbot files are mostly the same excluding the minor adjustments to suit the different controllers)
run_stim_rnn.py: This is the file where you can run simulations for specific agents. It is also the file used to gather data for controller performance analysis.
model_decentral_2.pth: A pre-trained decentralized model. Incase the user would like to see the performance of the decentralized controller, they can initiate this in the run_stim_rnn file without training any models.
model_central_3.pth: A pre-trained centralized model. Incase the user would like to see the performance of the centralized controller, they can initiate this in the run_stim_rnn file without training any models.
research_paper_thesis_final.pdf: The research paper explaining this project.

