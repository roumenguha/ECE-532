close all

x0 = zeros(1,100)';
x0(7) = 1.3;x0(27) = 1.3;x0(32) = 1.7;x0(68) = 2;x0(88) = 1.2;
h = [1 2 3 4 3 2 1]/16;
y = conv(x0,h);
y = y+0.05*randn(size(y));
N = 100;
H = convmtx(h',N);
lambda = 0.1;


x = ista(H, y, lambda);
figure,plot(x,'*-r'),title('estimated signal');
norm(x)

x = LassoIterativeSoftThresholding(H, y, lambda);
figure,plot(x,'*-r'),title('estimated signal');
norm(x)

x = hw6_532_fall17_ista(H, y, lambda);
figure,plot(x,'*-r'),title('estimated signal');
norm(x)