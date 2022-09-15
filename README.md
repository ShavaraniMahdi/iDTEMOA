# iDTEMOA
The implementation for iDTEMOA
Parameters of the class DTEMOA:
dm: an instance of MachineDM class which is a virtual DM and used for ranking solutions in the interactions. Any arbitrary UF can be used, if desired.
total_gen: Total number of generations
gen = Generations before the first interaction
 cr = 0.95, eta_c = 10., m = 0.01, eta_m = 50. : Parameters of NAGAII, defaulted to the original values.
 seed = random seed number
 geni= generations between subsequent interactions
 interactions=Total number of interaction
 sampleSize= number of solutions ranked  by the DM in each interaction
 
