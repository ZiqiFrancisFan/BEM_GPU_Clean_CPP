clear; close all; clc;
%%
a = 1;
r = 1:0.2:3;
rho = 1.2041;
c = 343.21;

f = 171.5;
k = 2*pi*f/c;

p = a^2./r*i*rho*2*pi*f/(1+i*k*a).*exp(-i*k*(r-a));

