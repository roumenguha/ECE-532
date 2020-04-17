function f = convt(h,g)
% f = convt(h,g);
% Transpose convolution: f = H' g
Nh = length(h);
Ng = length(g);
f = conv(h(Nh:-1:1), g);
f = f(Nh:Ng);