clc; clear all; close all;
SNRdB = [-10,-5,0,5,10];
SNR = 10.^(SNRdB./10);
N = 20;
M = 40000;
Pfa_theoretical = 0.0001:0.05:0.9999;
Pfa_simulated = zeros(length(SNR),N);

Pd_theoretical= zeros(length(SNR),length(Pfa_theoretical));
Pd_simulated = zeros(length(SNR),N);
false_alarm = zeros(1,M);
detected = zeros(1,M);
sigma2 = 1;% sigma^2
gamma = linspace(-5,5,N);
for ix = 1:length(SNR)
    s = sqrt(SNR(ix)).*[1;-1;1;-1]; 
    % H0 is true
    
    y = randn(length(s),M);
    for iy = 1:N
        false_alarm = (s'*y) > gamma(iy);
        Pfa_simulated(ix,iy) = sum(false_alarm,'all')/M;
    end
    %H1 is true
    y = s + randn(length(s),M);
    for iy = 1:N
        detected = (s'*y) > gamma(iy);
        Pd_simulated(ix,iy) = sum(detected,'all')/M;
    end
    Pd_theoretical(ix,:) = qfunc(qfuncinv(Pfa_theoretical) - norm(s)/sigma2);
end
plot(Pfa_theoretical, Pd_theoretical(1,:),'r -','linewidth',2.0 );
hold on; grid on; axis tight;
plot(Pfa_theoretical, Pd_theoretical(2,:),'g -','linewidth',2.0 );
hold on; grid on; axis tight;
plot(Pfa_theoretical, Pd_theoretical(3,:),'b -','linewidth',2.0 );
hold on; grid on; axis tight;
plot(Pfa_theoretical, Pd_theoretical(4,:),'c -','linewidth',2.0 );
hold on; grid on; axis tight;
plot(Pfa_theoretical, Pd_theoretical(5,:),'m -','linewidth',2.0 );
hold on; grid on; axis tight;
%simulated
scatter(Pfa_simulated(1,:), Pd_simulated(1,:),'o', 'MarkerFaceColor', 'r');
hold on; grid on; axis tight;
scatter(Pfa_simulated(2,:), Pd_simulated(2,:),'o', 'MarkerFaceColor', 'g')
hold on; grid on; axis tight;
scatter(Pfa_simulated(3,:), Pd_simulated(3,:),'o', 'MarkerFaceColor', 'b')
hold on; grid on; axis tight;
scatter(Pfa_simulated(4,:), Pd_simulated(4,:),'o', 'MarkerFaceColor', 'c')
hold on; grid on; axis tight;
scatter(Pfa_simulated(5,:), Pd_simulated(5,:),'o', 'MarkerFaceColor', 'm')
legend('Theoretical SNR = -10 dB','Theoretical SNR = -5 dB','Theoretical SNR = 0 dB','Theoretical SNR = 5 dB','Theoretical SNR = 10 dB','Simulated SNR = -10 dB','Simulated SNR = -5 dB','Simulated SNR = 0 dB','Simulated SNR = 5 dB','Simulated SNR = 10 dB')
legend('Location','southeast')
xlabel('Probability of False Alarm');
ylabel('Probability of Detection');
title('Simulated and Theoretical ROC curves ')
 