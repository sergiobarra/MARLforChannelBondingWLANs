clear 
close all
clc

seed_1992 = [1 0.947 0.3256 0.086] .*50 ;
seed_2042 = [0.0296	0.8748	0.615	0.6744].*50;
seed_2058 = [0.078	0.9804	0.004	0.9504].*50;
end_thr = [1 1 1 1].*50;
w1 = 0.5; 
w2 = 0.25;
x = [1 2 3 4];
Y_AXIS = [0, 51];
X_TICK_LABEL = {'A','B','C','D'};

figure
subplot(1,3,1)
bar(x,end_thr,w1,'FaceColor',[0.2 0.2 0.5])
hold on
bar(x,seed_1992,w2,'FaceColor',[0 0.7 0.7])
ylim(Y_AXIS)
grid on
xticklabels(X_TICK_LABEL)

subplot(1,3,2)
bar(x,end_thr,w1,'FaceColor',[0.2 0.2 0.5])
hold on
bar(x,seed_2042,w2,'FaceColor',[0 0.7 0.7])
ylim(Y_AXIS)
grid on
xticklabels(X_TICK_LABEL)

subplot(1,3,3)
bar(x,end_thr,w1,'FaceColor',[0.2 0.2 0.5])
hold on
bar(x,seed_2058,w2,'FaceColor',[0 0.7 0.7])
ylim(Y_AXIS)
grid on
xticklabels(X_TICK_LABEL)





Y_AXIS = [0, 1.00];

figure
bar(x,end_thr./50,w1,'FaceColor',[0.2 0.2 0.5])
hold on
bar(x,seed_1992./50,w2,'FaceColor',[0 0.7 0.7])
ylim(Y_AXIS)
grid on
xticklabels(X_TICK_LABEL)
yyaxis right
ylabel('thr. [Mbps]')

figure
bar(x,end_thr./50,w1,'FaceColor',[0.2 0.2 0.5])
hold on
bar(x,seed_2058./50,w2,'FaceColor',[0 0.7 0.7])
ylim(Y_AXIS)
grid on
xticklabels(X_TICK_LABEL)
yyaxis right
ylabel('thr. [Mbps]')

