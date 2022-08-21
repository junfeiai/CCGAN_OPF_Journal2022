%This is the matlab code for verification of the feasibility of OPF solutions
%IEEE case 300 does not have information for branch flow limits, we set all line constraints to be mpc.branch(:,6)=1000
filename = 'case300'; %case118,1354
mpc = loadcase(filename);
mpc.branch(:,6)=1000;
%generated solution by CCGAN
p_supply= csvread('p_supply.csv');
vm_gen = csvread('solutionv_list.csv');
pq_demand= csvread('demand_list.csv');
bus_number = demand_mat_size(2);
gen_number = supply_mat_size(2);
p_supply = p_supply(:,2:gen_number+1)*1.0;
p_demand=pq_demand(:,2:bus_number+1);
q_demand=pq_demand(:,bus_number+2:bus_number*2+1);
demand_mat_size = size(p_demand);
supply_mat_size = size(p_supply);
sample_size = demand_mat_size(1)-1;

success_number = 0;
l=0;
bad_ids=[];
for i=1:1000
    p_demand_i = p_demand(i+1,:);
    q_demand_i = q_demand(i+1,:);
    p_supply_i = p_supply(i+1,:);
    p_sum = sum(p_demand_i,2);
    q_sum = sum(q_demand_i,2);
    p_supply_sum = sum(p_supply_i,2);
    l=l+1;
    vm_gen_i = vm_gen(i+1,:);
    mpc.bus(:,3) = p_demand_i(1:bus_number)*100;
    mpc.bus(:,4) = q_demand_i(1:bus_number)*100;
    mpc.gen(:,2) = p_supply_i(1:gen_number)*100;
    pp=0;
    for bus_i=1:bus_number
        if  ismember(mpc.bus(bus_i,1),mpc.gen(:,1)) 
            mpc.bus(bus_i,8) = vm_gen_i(bus_i);
            pp=pp+1;
        end
    end
    try 
        [result,success]=runpf(mpc,mpoption('pf.enforce_q_lims', 1,'pf.tol',1e-3,'pf.nr.max_it',10));
    catch
        
    end
end    
