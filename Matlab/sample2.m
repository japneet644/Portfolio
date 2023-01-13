clc; close all; clear all;
h = [1;1;1;1];
N = 8;
sigma2db = -10:1:10; 
sigma2 = 10.^(sigma2db./10);
NOfIter = 5000;
MSE1_avg = zeros(length(sigma2),1);
MSE2_avg = zeros(length(sigma2),1);
MSE1_theoretical_avg = zeros(length(sigma2),1);
MSE2_theoretical_avg = zeros(length(sigma2),1);
for iter = 1:NOfIter
    %case-1
    y = zeros(N,1);
    for j = 1:length(sigma2)
        X = randn(N,length(h));
        X = X./sqrt(trace(X'*X));%normalizing so that trace is 1
        for i=1:N
            y(i) = X(i,:)*h + sqrt(sigma2(j))*randn(1); %Y = x.T*h + v
        end
        ML_estimate1 = inv(X'*X)*X'*y;
        MSE1_avg(j) = MSE1_avg(j) + norm(h- ML_estimate1)^2/NOfIter;
        MSE1_theoretical_avg(j) = MSE1_theoretical_avg(j) + sigma2(j)*trace(inv(X'*X))/NOfIter;
    end
    
    % case 2
    y = zeros(N,1);
    for j = 1:length(sigma2)
        Z = dftmtx(N);
        X2 = Z(:,1:4);%eye(N,length(h));
        X2 = X2./sqrt(trace(X2'*X2));%normalizing so that trace is 1
        for i=1:N
            y(i) = X2(i,:)*h + sqrt(sigma2(j))*randn(1);
        end
        ML_estimate2 = inv(X2'*X2)*X2'*y;
        MSE2_avg(j) = MSE2_avg(j) + norm(h-ML_estimate2)^2/NOfIter;
        MSE2_theoretical_avg(j) = MSE2_theoretical_avg(j) + sigma2(j)*trace(inv(X2'*X2))/NOfIter;
    end
end

figure(1)
semilogx(sigma2,MSE1_avg, 'b-s','linewidth',2.0 );
hold on;
semilogx(sigma2,MSE2_avg, 'r-s','linewidth',2.0 );
hold on;
semilogx(sigma2,MSE1_theoretical_avg, 'b-.','linewidth',2.0 );
hold on;
semilogx(sigma2,MSE2_theoretical_avg, 'r-.','linewidth',2.0 );
xlabel('Sigma^2', 'FontSize', 14);
ylabel('MSE', 'FontSize', 14);
title('MSE', 'FontSize', 14);
legend('Case-1','Case-2','Case-1-theoretical','Case-2-theoretical')
grid on;