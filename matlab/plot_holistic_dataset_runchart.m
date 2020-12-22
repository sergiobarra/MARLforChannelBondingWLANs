% ---
% This file generates the Figure of paper TO_BE_PUBLISHED
% More info at https://github.com/sergiobarra/MARLforChannelBondingWLANs

clc
close all
clear

%%
SEEDS = 1992:2091;
NUM_ITERATIONS = 200;

epsilon_c_min = zeros(length(SEEDS),NUM_ITERATIONS);
epsilon_c_mean = zeros(length(SEEDS),NUM_ITERATIONS);
contextual_epsilon_02contexts_c_min = zeros(length(SEEDS),NUM_ITERATIONS);
contextual_epsilon_02contexts_c_mean = zeros(length(SEEDS),NUM_ITERATIONS);
contextual_epsilon_24contexts_c_min = zeros(length(SEEDS),NUM_ITERATIONS);
contextual_epsilon_24contexts_c_mean = zeros(length(SEEDS),NUM_ITERATIONS);

qlearning_02states_c_min = zeros(length(SEEDS),NUM_ITERATIONS);
qlearning_02states_c_mean = zeros(length(SEEDS),NUM_ITERATIONS);
qlearning_24states_c_min = zeros(length(SEEDS),NUM_ITERATIONS);
qlearning_24states_c_mean = zeros(length(SEEDS),NUM_ITERATIONS);

fprintf('Generating reward runchart of paper TO_BE_PUBLISHED...\n')

for seed_ix = 1:length(SEEDS)

    % epsilon
    filename = ['../sim_output/egreedy/epsilon_' num2str(SEEDS(seed_ix)) '_summary.csv'];
    M = csvread(filename);
    epsilon_c_min(seed_ix,:) = M(:,11);
    epsilon_c_mean(seed_ix,:) = M(:,12);
    
    % contextual e-greedy 2-contexts
    filename = ['../sim_output/contextual_2contexts/contextual_epsilon_02contexts_' num2str(SEEDS(seed_ix)) '_summary.csv'];
    M = csvread(filename);
    contextual_epsilon_02contexts_c_min(seed_ix,:) = M(:,11);
    contextual_epsilon_02contexts_c_mean(seed_ix,:) = M(:,12);
    
    % contextual e-greedy 24-contexts
    filename = ['../sim_output/contextual_24contexts/contextual_epsilon_24contexts_' num2str(SEEDS(seed_ix)) '_summary.csv'];
    M = csvread(filename);
    contextual_epsilon_24contexts_c_min(seed_ix,:) = M(:,11);
    contextual_epsilon_24contexts_c_mean(seed_ix,:) = M(:,12);
    
    % q-learning 2-states (alpha=0.8, gamma = 0.2)
    filename = ['../sim_output/q_learning_2states_alpha0.8_gamma0.2/qlearning_02states_' num2str(SEEDS(seed_ix)) '_summary.csv'];
    M = csvread(filename);
    qlearning_02states_c_min(seed_ix,:) = M(:,11);
    qlearning_02states_c_mean(seed_ix,:) = M(:,12);
    
    % q-learning 24-states (alpha=0.8, gamma = 0.2)
    filename = ['../sim_output/q_learning_24states_alpha0.8_gamma0.2/qlearning_24states_' num2str(SEEDS(seed_ix)) '_summary.csv'];
    M = csvread(filename);
    qlearning_24states_c_min(seed_ix,:) = M(:,11);
    qlearning_24states_c_mean(seed_ix,:) = M(:,12);

%     % q-learning 2-states (alpha=0.9, gamma = 0.1)
%     filename = ['../sim_output/q_learning_2states_alpha0.9_gamma0.1/qlearning_02states_alpha0.9_gamma0.1_' num2str(SEEDS(seed_ix)) '_summary.csv'];
%     M = csvread(filename);
%     qlearning_02states_c_min(seed_ix,:) = M(:,11);
%     qlearning_02states_c_mean(seed_ix,:) = M(:,12);
%     
%     % q-learning 24-states (alpha=0.9, gamma = 0.1)
%     filename = ['../sim_output/q_learning_24states_alpha0.9_gamma0.1/qlearning_24states_alpha0.9_gamma0.1_' num2str(SEEDS(seed_ix)) '_summary.csv'];
%     M = csvread(filename);
%     qlearning_24states_c_min(seed_ix,:) = M(:,11);
%     qlearning_24states_c_mean(seed_ix,:) = M(:,12);

%     % q-learning 2-states (alpha=0.95, gamma = 0.05)
%     filename = ['../sim_output/q_learning_2states_alpha0.95_gamma0.05/qlearning_02states_alpha0.95_gamma0.05_' num2str(SEEDS(seed_ix)) '_summary.csv'];
%     M = csvread(filename);
%     qlearning_02states_c_min(seed_ix,:) = M(:,11);
%     qlearning_02states_c_mean(seed_ix,:) = M(:,12);
%     
%     % q-learning 24-states (alpha=0.95, gamma = 0.05)
%     filename = ['../sim_output/q_learning_24states_alpha0.95_gamma0.05/qlearning_24states_alpha0.95_gamma0.05_' num2str(SEEDS(seed_ix)) '_summary.csv'];
%     M = csvread(filename);
%     qlearning_24states_c_min(seed_ix,:) = M(:,11);
%     qlearning_24states_c_mean(seed_ix,:) = M(:,12);

end

LINEWIDTH_MEAN = 3;
Y_LIM=[0 1];


figure
subplot(2,5,1)
hold on
plot(epsilon_c_min')
plot(mean(epsilon_c_min),'linewidth',LINEWIDTH_MEAN)
grid on
xlabel('iteration')
ylabel({'norm. cum. reward';'worse'})
title('e-greedy')
ylim(Y_LIM)

subplot(2,5,2)
hold on
plot(contextual_epsilon_02contexts_c_min')
plot(mean(contextual_epsilon_02contexts_c_min),'linewidth',LINEWIDTH_MEAN)
title('context(2)')
grid on
xlabel('iteration')
ylim(Y_LIM)

subplot(2,5,3)
hold on
plot(contextual_epsilon_24contexts_c_min')
plot(mean(contextual_epsilon_24contexts_c_min),'linewidth',LINEWIDTH_MEAN)
title('context(24)')
grid on
xlabel('iteration')
ylim(Y_LIM)

subplot(2,5,4)
hold on
plot(qlearning_02states_c_min')
plot(mean(qlearning_02states_c_min),'linewidth',LINEWIDTH_MEAN)
title('Q-learning(2)')
grid on
xlabel('iteration')
ylim(Y_LIM)

subplot(2,5,5)
hold on
plot(qlearning_24states_c_min')
plot(mean(qlearning_24states_c_min),'linewidth',LINEWIDTH_MEAN)
title('Q-learning(24)')
grid on
xlabel('iteration')
ylim(Y_LIM)

subplot(2,5,6)
hold on
plot(epsilon_c_mean')
plot(mean(epsilon_c_mean),'linewidth',LINEWIDTH_MEAN)
grid on
xlabel('iteration')
ylabel({'norm. cum. reward';'mean'})
ylim(Y_LIM)

subplot(2,5,7)
hold on
plot(contextual_epsilon_02contexts_c_mean')
plot(mean(contextual_epsilon_02contexts_c_mean),'linewidth',LINEWIDTH_MEAN)
grid on
xlabel('iteration')
ylim(Y_LIM)

subplot(2,5,8)
hold on
plot(contextual_epsilon_24contexts_c_mean')
plot(mean(contextual_epsilon_24contexts_c_mean),'linewidth',LINEWIDTH_MEAN)
grid on
xlabel('iteration')
ylim(Y_LIM)


subplot(2,5,9)
hold on
plot(qlearning_02states_c_mean')
plot(mean(qlearning_02states_c_mean),'linewidth',LINEWIDTH_MEAN)
grid on
xlabel('iteration')
ylim(Y_LIM)

subplot(2,5,10)
hold on
plot(qlearning_24states_c_mean')
plot(mean(qlearning_24states_c_mean),'linewidth',LINEWIDTH_MEAN)
grid on
xlabel('iteration')
ylim(Y_LIM)

%%%%%

LINEWIDTH_MEAN = 2;

figure
subplot(1,2,1)
hold on
plot(mean(epsilon_c_min),'-k','linewidth',LINEWIDTH_MEAN)
plot(mean(contextual_epsilon_02contexts_c_min),'-b','linewidth',LINEWIDTH_MEAN)
plot(mean(contextual_epsilon_24contexts_c_min),'-.b','linewidth',LINEWIDTH_MEAN)
plot(mean(qlearning_02states_c_min),'-r','linewidth',LINEWIDTH_MEAN)
plot(mean(qlearning_24states_c_min),'-.r','linewidth',LINEWIDTH_MEAN)
title('worse')
grid on
ylabel('norm. cum. reward')
ylim(Y_LIM)

subplot(1,2,2)
hold on
plot(mean(epsilon_c_mean),'-k','linewidth',LINEWIDTH_MEAN)
plot(mean(contextual_epsilon_02contexts_c_mean),'-b','linewidth',LINEWIDTH_MEAN)
plot(mean(contextual_epsilon_24contexts_c_mean),'-.b','linewidth',LINEWIDTH_MEAN)
plot(mean(qlearning_02states_c_mean),'-r','linewidth',LINEWIDTH_MEAN)
plot(mean(qlearning_24states_c_mean),'-.r','linewidth',LINEWIDTH_MEAN)
title('mean')
grid on
ylim(Y_LIM)

legend('e-greedy','contextual(2)','contextual(24)','q-learning(2)','q-learning(24)')


fprintf('Figure plotted!\n')
