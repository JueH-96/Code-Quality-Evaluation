import sys

def ctz(x):
    return (x & -x).bit_length() - 1

N = int(sys.stdin.readline().strip())
print(ctz(N))