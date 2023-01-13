clc; clear all; close all;
SNRdB = [-10,-5,0,5,10];
SNR = 10.^(SNRdB./10);
N = 10000;

Pfa = 0:0.1:1;
Pd_theoretical = zeros(length(Pfa));
Pd_simulated = zeros(length(Pfa));

for ix = 1:length(SNR)
    s = sqrt(SNR(ix)).*[1,-1,1,-1]; 
    gamma = norm(s)*qfuncinv(Pfa(ix));
    % H0 is true
    for iy = 1:N
        
    end
    for iy = 1:N
        
    end
end