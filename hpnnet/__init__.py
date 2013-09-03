
import os
import psutil
p = psutil.Process(os.getpid())
all_cpus = list(range(psutil.NUM_CPUS))
if p.get_cpu_affinity() != all_cpus:
    print 'Setting CPU AFFINITY to %s' % str(all_cpus)
    p.set_cpu_affinity(all_cpus)
