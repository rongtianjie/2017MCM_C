clc;
clear all;
close all;

B=3;             % The number of the lanes
plazalength=50;  % The length of the simulating highways
h=NaN;           % h is the handle of the image


[plaza,v]=create_plaza(B,plazalength);
h=show_plaza(plaza,h,0.1);

iterations=1000;    
probc=0.1;          
probv=[0.1 1];      
probslow=0.3;       
Dsafe=1;            
VTypes=[1,2];       
[plaza,v,vmax]=new_cars(plaza,v,probc,probv,VTypes);

size(find(plaza==1))
PLAZA=rot90(plaza,2);
h=show_plaza(PLAZA,h,0.1);
for t=1:iterations;
    size(find(plaza==1))
    PLAZA=rot90(plaza,2);
    h=show_plaza(PLAZA,h,0.1);
    [v,gap,LUP,LDOWN]=para_count(plaza,v,vmax);
    [plaza,v,vmax]=switch_lane(plaza,v,vmax,gap,LUP,LDOWN);
    [plaza,v,vmax]=random_slow(plaza,v,vmax,probslow);
    [plaza,v,vmax]=move_forward(plaza,v,vmax);
end




