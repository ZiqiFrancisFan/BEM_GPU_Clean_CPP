clear; close all; clc;
%%
a = 1;
r = 1:0.2:3;
rho = 1.2041;
c = 343.21;

f = 171.5;
k = 2*pi*f/c;

p = a^2./r*i*rho*2*pi*f/(1+i*k*a).*exp(-i*k*(r-a));

%%
fileID = fopen('sphere171Hz_NoCHIEF','r');
if(fileID == -1)
    printf('Failed to open file HRTFs');
    return;
end
format = '';
s = '(%f,%f) ';
for i=1:length(r)
    format = [format,s];
end

pressure_nc = zeros(size(r,1),size(r,2));
for i=1:size(r,1)
    tline = fgetl(fileID);
    a = sscanf(tline,format);
    for j=1:size(r,2)
        x = a(2*j-1);
        y = a(2*j);
        pressure_nc(i,j) = x+complex(0,1)*y;
    end
end
fclose(fileID);

%%
fileID = fopen('sphere171Hz_CHIEF','r');
if(fileID == -1)
    printf('Failed to open file HRTFs');
    return;
end
format = '';
s = '(%f,%f) ';
for i=1:length(r)
    format = [format,s];
end
pressure_c = zeros(size(r,1),size(r,2));
for i=1:size(r,1)
    tline = fgetl(fileID);
    a = sscanf(tline,format);
    for j=1:size(r,2)
        x = a(2*j-1);
        y = a(2*j);
        pressure_c(i,j) = x+complex(0,1)*y;
    end
end
fclose(fileID);

%%
fig = figure;
ax = axes(fig);
plot(ax,r,abs(p),'-*b');
hold(ax,'on');
plot(ax,r,abs(pressure_nc),'-+g');
hold(ax,'on');
plot(ax,r,abs(pressure_c),'-or');
legend(ax,'Ground truth','Simulation without CHIEF','Simulation with CHIEF');
xlabel(ax,'Radius (m)');
ylabel(ax,'Magnitude');
title(ax,'Magnitude response of a pulsating sphere');

fig = figure;
ax = axes(fig);
plot(ax,r,angle(p)/pi,'-*b');
hold(ax,'on');
plot(ax,r,angle(pressure_nc)/pi,'-+g');
hold(ax,'on');
plot(ax,r,angle(pressure_c)/pi,'-or');
legend(ax,'Ground truth','Simulation without CHIEF','Simulation with CHIEF');
xlabel(ax,'Radius (m)');
ylabel(ax,'Phase (\pi)');
title(ax,'Phase response of a pulsating sphere');


