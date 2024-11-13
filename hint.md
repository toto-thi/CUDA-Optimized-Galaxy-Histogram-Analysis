
# Running on HPC (or your own GPU) with"

   srun -p gpu -n 1 -t 10:00 --mem=1G -e err.txt -o out.txt time ./galaxy data_100k_arcmin.dat flat_100k_arcmin.dat result.out

the program finishes in 2 seconds: 0.55user 0.35system 0:00.91elapsed 98%CPU
On an Tesla V100-PCIE-16GB, the run time was 0.78 seconds.
