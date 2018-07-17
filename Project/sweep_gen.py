import numpy as np
variables = ["slew", "load", "rload", "freq"]
smallest_slew = 1e-12
largest_slew = 1e-10
smallest_load = 1e-14
largest_load = 1e-12
smallest_freq = 1e7
largest_freq = 2e8 # don't change this, it's the actual max for this circuit
smallest_r_fixture = 50
largest_r_fixture = 150
#slews = [1e-12, 4e-12, 8e-12, 1e-11, 4e-11, 8e-11, 1e-10]
#loads = [1e-14, 4e-14, 8e-14, 1e-13, 4e-13, 8e-13, 1e-12]
slews = [1e-12, 1e-10]
loads = [1e-13, 1e-12]
freqs = [4e7, 8e7, 1e8]
rloads = [50, 100, 150]

def write_head(fd):
    fd.write(".data arc_data\n")

def write_variables(fd):
    s = "+ "
    s2 = " ".join(variables)
    fd.write(s+s2+"\n")

def write_end(fd):
    fd.write(".enddata\n")

def gen_fixed(filename):
    with open(filename, 'w') as fd:
        write_head(fd)
        write_variables(fd)
        for slew in slews:
            for load in loads:
                for rload in rloads:
                    for freq in freqs:
                        fd.write("+ %e  %e  %e  %e\n"%(slew, load, rload,  freq))
        write_end(fd)

def gen_linear(filename, config=None):
    slewNo = 50
    loadNo = 50
    freqNo = 2
    rNo = 3
    if config:
        slewNo = config['slewNo']
        loadNo = config['loadNo']
        freqNo = config['freqNo']
    with open(filename, 'w') as fd:
        write_head(fd)
        write_variables(fd)
        slew = smallest_slew
        for slew in np.linspace(smallest_slew, largest_slew, num=slewNo):
            for load in np.linspace(smallest_load, largest_load, num=loadNo):
                for rload in np.linspace(smallest_r_fixture, largest_r_fixture, num=rNo):
                    for freq in np.linspace(smallest_freq, largest_freq, num=freqNo):
                        fd.write("+ %e  %e  %e  %e\n"%(slew, load, rload, freq))
        write_end(fd)

if __name__ == "__main__":
    filename = 'sweep.inc'
    #gen_linear(filename)
    gen_fixed(filename)
