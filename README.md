# Multi-Agent Reinforcement Learning (MARL) for Spectrum Allocation in Dynamic Channel Bonding WLANs

**Scientific papers using this MARL framework**: 
- *To be announced.

### Author
* [Sergio Barrachina-Muñoz](https://github.com/sergiobarra)

### Project description

Adaptive spectrum allocation in dynamic channel bonding where different BSS's may change their primary and max. bandwidth allocation generates complex and very dynamic environments where heuristic-based algorithms normally fail to reach satisfactory configurations. In this regard, we study multi-agent RL variations to cope with the task of letting uncoordinated BSS's rapidly find satisfactory spectrum configurations.

<img src="https://github.com/sergiobarra/MARLforChannelBondingWLANs/blob/master/images/marl_diagram.png" alt="MARL diagram"
	title="MARL diagram" width="300" />


### Repository description
This repository contains the Jupyter Notebooks to simulate MARL behavior in a multi-agent WLAN with uncoordinated BSS's. It contains a variety of classes to generate RL instances such as different MAB exploration algorithms (e.g., e-greedy, exploration-first, Thompson Sampling, UCB1, etc.), contextual MABs, and Q-learning. Including new RL models to the project should be straightforward. The RL algorithms should evolve given a holistic dataset where all the possible system (or global) configurations are pre-simulated. That is, the framework maps the global configuration reached through the MARL interaction, queries the resulting performance of each BSS in the dataset at that system status, and feeds the corresponding value to each learning instance.

* Main file: ```main.ipynb```
* Custom RL models file: ```rl_models.ipynb```

### 4-BSS's selfcontained dataset v1 (2020)

In our research publications, we work with a holistic, self-contained, toy scenario dataset, resulting from simulating all the possible configuration combinations of a particular deployment of 4 potentially overlapping BSS's with dynamic channel bonding capabilities. The idea of simulating every configuration combination is to know the optimal one for every combination of traffic loads.

We study the deployment illustrated below, consisting of 4 BSS's (with one AP and one STA each) in a system of 4 20-MHz channels, where APs send downlink UDP traffic to their STAs. Traffic load is quantized and can take three values (20, 50, and 150 Mbps). As for the action attributes, the **primary channel** can take any of the 4 channels in the system, and the **maximum bandwidth** is allowed to be set to 1, 2, and 4 20-MHz channels, fulfilling the IEEE 802.11ac/ax channelization restrictions for 20, 40, and 80 MHz bandwidths. So, the total number of global statuses when considering all BSS to have an agent-empowered AP raises to (4 x 3 x 3)^4 = 1,679,616.

<img src="https://github.com/sergiobarra/MARLforChannelBondingWLANs/blob/master/images/toy_scenario_deployment.png" alt="Toy scenario deployment"
	title="Toy scenario deployment" width="400" />

The  interference  matrix  elements  in  the Figure above indicate the maximum  bandwidth  in  MHz  that  causes  two  APs  tooverlap. So,this deployment is complex in the sense that multiple different  one-to-one  overlaps  The interference matrix elements in Figure above indicate the maximum bandwidth in MHz that causes two APs to overlap. So, this deployment is complex in the sense that multiple different one-to-one overlaps appear depending on the distance and the transmission bandwidth in use, leading to exposed and hidden node situations hard to prevent beforehand. For instance, AP_A and AP_C can only overlap when using 20 MHz, whereas AP_A and AP_D do always overlap regardless of the bandwidth because of their proximity. Instead, AP_A and AP_B overlap whenever 40 MHz (or 20 MHz) bandwidth is used.

The dataset is generated by simulating every possible combination of spectrum configuration (primary and max bandwidth) and traffic load of each BSS. This combination is what we call *status*. We use the Komondor wireless simulator to perform the simulation flow diagram shown below.

<img src="https://github.com/sergiobarra/MARLforChannelBondingWLANs/blob/master/images/simulation_flow_diagram.png" alt="Simulation flow diagram"
	title="Simulation flow diagram" width="400" />

The 4-BSS's self-contained dataset for spectrum management in WLANs can be found [here](https://www.upf.edu/web/wnrg/wn-datasets)

### Contribute

If you want to contribute, please contact to [sergio.barrachina@upf.edu](sergio.barrachina@upf.edu)
