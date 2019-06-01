import logging

class Config:

    def __init__(self, is_multithread=True):
        self.is_multithread = is_multithread

        # Logging params
        self.Logger_Level = logging.INFO
        self.Logger_Format = '%(asctime)s %(message)s'

        # Game environment params
        self.Width = 7
        self.Height = 6

        # network parameters
        self.Network_Metadata = [{'filters':42, 'kernel_size':(4, 4)},
                                 {'filters':42, 'kernel_size':(4, 4)},
                                 {'filters':42, 'kernel_size':(4, 4)},
                                 {'filters':42, 'kernel_size':(4, 4)},
                                 {'filters':42, 'kernel_size':(4, 4)},
                                 {'filters':42, 'kernel_size':(4, 4)}]
        self.Reg_Const = 0.01
        self.Learning_Rate = 0.00025
        self.Root_Path = 'C:/Users/morel/Machine Learning/Reinforcement Learning/AlphaGO Zero on Connect 4'

        # monte carlo tree search parameters
        self.Cpuct = 1
        self.Temperature = 0.01 # this value should be in the range [0, 1]. The lower the value, the certainer the probability
        self.Dir_Epsilon = 0.25 # Dirichlet noise params
        self.Dir_Alpha = 0.03 # Dirichlet noise params

        # training parameters
        if self.is_multithread:
            self.Episode_num = 0    # ???
            self.Epochs_Num = 2     # Number of epochs (pass through the batch size) while fitting the neural network
            self.Batch_Size = 16
        else:
            self.Episode_num = 100 # not used by multithreading
            self.Epochs_Num = 10
            self.Batch_Size = 16

        self.MCTS_Num = 100  # Number of iterations of a MCTS
        self.Memory_Size = 20000  # Number of steps ((state, action_probs, value, player) quadruples) that the memory will hold
        self.Iteration_Num = 500  # Number of updates of the best network during comparison thread
        self.Sample_Size = 64  # Number of steps used for one iteration of the fit function (same as data size, i think)
        self.Compete_Game_Num = 30  # Number of games two networks are competing on
        self.Best_Network_Threshold = 0.6  # Ratio of wins a network has to have to be considered the best
        self.Validation_Split = 0.1  # Ration of cross validation data

        # multithreading parameters
        self.Fit_Interval = 3   # waiting time after done fitting the network
        self.Comparison_Interval = 5    # the comparison thread sleeps during this time to wait untils the self play thread gathers enough data
        self.Comparison_Long_Wait = 1200    # the comparidon thread sleeps during this time to give time to the fit thread to fit the network
        self.Min_Memory_Size_Before_Fit = int(self.Memory_Size * 0.1)   # Number of steps to have in memory before starting to fit
        self.New_Best_Network_Memory_Clean_Rate = 1 # Percentage of data that should be erased from memory when a new network is created (after beating best network)

        # Test parameters
        self.Test_MCTS_Num = 100    # number of iterations of tree search when two networks will be competing against each other