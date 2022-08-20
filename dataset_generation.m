%This is the matlab code for generating valid IEEE118 grid power flow data
%This code can also be used to generate OPF solutions for IEEE 118 by commenting line 58,59,62; and use line 63.
%Original Matpower grid 118 does not have line constraint, However, a branch flow limits template has been released by IIT, 
%“Index of Data Illinois Institute of Technology.” [Online]. Available: http://motor.ece.iit.edu/data/. 
%Load is sampled in plus/minus 20 percent
%DERs are considered: wind and solar: line 37-44

filename = 'case118';
mpc = loadcase(filename);
line_constraint = csvread('ROSCUC_118.csv');
mpc.branch(:,6) = line_constraint(:,5);
solar_der_flag = csvread('solar_der_flag.csv');
solar_der_flag = solar_der_flag(2:119,2);
wind_der_flag = csvread('wind_der_flag.csv');
wind_der_flag = wind_der_flag(2:119,2);
number_of_panels = 10000;
default_demand_p = mpc.bus(:,3);
default_demand_q = mpc.bus(:,4);
grid_size_full = size(default_demand_p);
grid_size = grid_size_full(1,1);
datapoints_list=[];
conditions_list=[];
flag=0;
bus_has_load_p = default_demand_p>0;
bus_has_load_q = default_demand_q>0;
success_number = 0;
while flag==0
    mpc = loadcase(filename);
    %Random perturbation on load P
    rdm_per_p = ones(grid_size,1)*(-0.2)+rand(grid_size,1)*0.4;
    rdm_per_q = ones(grid_size,1)*(-0.2)+rand(grid_size,1)*0.4;
    new_p = default_demand_p.*(1+rdm_per_p).*bus_has_load_p;
    new_q = default_demand_q.*(1+rdm_per_q).*bus_has_load_q;
    mpc.bus(:,3) = new_p;
    mpc.bus(:,4) = new_q;
    
    %Wind and solar generation as perturbations to loads
    wind_wb_var = wblrnd(2,5,118,1);
    wind_power_var = wind_wb_var.^3*0.5*1.25*1800/1000000;
    wind_power_der = 100*wind_der_flag.*wind_power_var;
    solar_wb_var = wblrnd(2,5,118,1);
    solar_power_var = solar_wb_var*number_of_panels/1000000;
    solar_power_der = 100*solar_der_flag.*solar_power_var;
    mpc.bus(:,3) = mpc.bus(:,3)-wind_power_der-solar_power_der;
    
    %Random active power generation sampling
    new_pg=mpc.gen(:,5)+rand(54,1).*(mpc.gen(:,4)-mpc.gen(:,5));
    mpc.gen(:,2) = new_pg;
    
    default_cost_c1 = mpc.gencost(:,6);
    default_cost_c2 = mpc.gencost(:,5);
    
    gen_size = size(default_cost_c1);

    new_c1 = mpc.gencost(:,6)*0.1+rand(gen_size).*mpc.gencost(:,6);
    new_c2 = mpc.gencost(:,5)*0.1+rand(gen_size).*mpc.gencost(:,5);
    
    mpc.gencost(:,6) = new_c1;
    mpc.gencost(:,5) = new_c2;
    %Trying the PF solver
    try 
        [result,success]=runopf(mpc,mpoption('pf.enforce_q_lims', 1,'opf.ac.solver', 'SDPOPF'));
        %[result,success]=runpf(mpc);
        if success==1
            [FV, PV] = checklimits(result);
            if ~isempty(FV.i)
                success=0;
            end
        end
    catch
        
    end
        
    if success==1 && all(result.gen(:,3)<=result.gen(:,4)) && all(result.gen(:,3)>=result.gen(:,5))
        success_number=success_number+1;
        datapoint = [result.bus(:,8).',result.bus(:,9).',result.gen(:,2).',result.gen(:,3).'];
        condition = [mpc.bus(:,3).',mpc.bus(:,4).'];
        datapoints_list = [datapoints_list;datapoint];
        conditions_list = [conditions_list;condition];
    end
    cl_size = size(conditions_list);
    if cl_size(1)>1000
        flag=1;
    end
end
