function [plaza,v,vmax]=move_forward(plaza,v,vmax);
    [L,W]=size(plaza);
    
    gap=zeros(L,W);
    for lanes=2:W-1;
        temp=find(plaza(:,lanes)==1); 
        nn=length(temp);
        for k=1:nn;
            i=temp(k);
            if(k==nn)
                gap(i,lanes)=L-(temp(k)-temp(1)+1);
                continue;
            end
            gap(i,lanes)=temp(k+1)-temp(k)-1;
        end
    end
%      for i=1:L;
%         for j=1:W;
%             if(plaza(i,j)~=1)
%                     v(i,j)=-88;
%                     gap(i,j)=-99;
%             end
%         end
%     end
%     v
%     gap
     for lanes=2:W-1;
         temp=find(plaza(:,lanes)==1);
         nn=length(temp);
         for k=1:nn;
             i=temp(k);
             if(v(i,lanes)<=gap(i,lanes))
                pos=mod(i+v(i,lanes)-1,L)+1;
             end
             if(v(i,lanes)>gap(i,lanes))
                pos=mod(i+gap(i,lanes)-1,L)+1;
             end
             if(pos~=i)
                plaza(pos,lanes)=1;
                v(pos,lanes)=v(i,lanes);
                vmax(pos,lanes)=vmax(i,lanes);
                 plaza(i,lanes)=0;
                 v(i,lanes)=0;
                 vmax(i,lanes)=0;
             end
         end
     end

end