h = 5;
N = 10;
sigma2 = 10^0;% sigma^2 in absolute value (not in Db)
iter = 100000;
hcap = zeros(iter,1);
for i=1:iter
    v = sigma2.*randn(N,1);
    hcap(i) = sum(h+v)/N;
end
nbins = 100;
figure(1)
ht = histogram(hcap,nbins);
ht.Normalization = 'pdf';
xlim([4, 6]);
xlabel('Estimated Value', 'FontSize', 14);
ylabel('PDF', 'FontSize', 14);
title('Histogram of Data N = 10', 'FontSize', 14);



figure(2)
N = 100;
hcap = zeros(iter,1);
for i=1:iter
    v = sigma2.*randn(N,1);
    hcap(i) = sum(h+v)/N;
end
nbins = 100;
figure(2)
ht = histogram(hcap,nbins);
ht.Normalization = 'pdf';
xlim([4, 6]);
xlabel('Estimated Value', 'FontSize', 14);
ylabel('PDF', 'FontSize', 14);
title('Histogram of Data N = 10', 'FontSize', 14);

