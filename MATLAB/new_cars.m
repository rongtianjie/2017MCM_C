function [plaza,v,vmax]=new_cars(plaza,v,probc,probv,VTypes);
    [L,W]=size(plaza);
%     plaza(2,3)=1;v(2,3)=1;vmax(2,3)=2;
%     plaza(3,3)=1;v(3,3)=1;vmax(3,3)=1; 
%     plaza(4,3)=1;v(4,3)=1;vmax(4,3)=1;
%     plaza(5,3)=1;v(5,3)=1;vmax(5,3)=2;
    vmax=zeros(L,W);
    for lanes=2:W-1;
        for i=1:L;
            if(rand<=probc)
                tmp=rand;
                plaza(i,lanes)=1;
                for k=1:length(probv)
                    if(tmp<=probv(k))
                        vmax(i,lanes)=VTypes(k);
                        v(i,lanes)=ceil(rand*vmax(i,lanes));
                        break;
                    end
                end
            end
        end
    end
    
    needn=ceil((W-2)*L*probc);
    number=size(find(vmax~=0),1);
    if(number<needn)
        while(number~=needn)
            i=ceil(rand*L);
            lanes=floor(rand*(W-2))+2;
            if(plaza(i,lanes)==0)
                plaza(i,lanes)=1;
                for k=1:length(probv)
                   if(tmp<=probv(k))
                       vmax(i,lanes)=VTypes(k);
                       v(i,lanes)=ceil(rand*vmax(i,lanes)); 
                       break;
                   end
                end
                number=number+1;
            end
        end
    end
    if(number>needn)
        temp=find(plaza==1);
        for k=1:number-needn;
            i=temp(k);
            plaza(i)=0;
            vmax(i)=0;
            v(i)=0;
        end
    end
end