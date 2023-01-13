clc;
close all;
n = 10;
% d = 1;
% m = 10;
% r = randi([1 2],n,1);
f = 1./linspace(1,n,n);
f(7:10)= 0;
Q = diag(f) ;
t = 100;

U1 = zeros(15,1);
for ko = 1:15
    U = zeros(n,n);
    for kx = 1:t
        P = randn(ko,n);
            X = pinv(P*Q*P');        
        U = U + (1/t).*P'*X*P;
    end
    U1(ko,1) = sum(abs(U - pinv(Q)).^2,'all');
    ko
    U
end
A = 1:15;
semilogy(A,U1(:,1),'LineWidth',1.5);
hold on;
% plot(A,U1(2,:),'LineWidth',1.5);
% hold on;
% plot(U1(3,:),'LineWidth',1.5);
% hold on;
% plot(A,U1(4,:),'LineWidth',1.5);
% hold on;
% plot(A,U1(5,:),'LineWidth',1.5);
% hold on;
% plot(U1(6,:),'LineWidth',1.5);
% hold on;
% plot(A,U1(7,:),'LineWidth',1.5);
% hold on;
% plot(U1(8,:),'LineWidth',1.5);
grid on;
axis tight;
xlabel('Sketch size')
ylabel('Error (in L2)')
% legend('m = 1','m=5','m=10', 'm=15','m=20','m=25','m=30');
title([' $$ || S^T(S Q S^T)^{-1} S - Q^{-1} ||^2$$ '],'interpreter','latex')
