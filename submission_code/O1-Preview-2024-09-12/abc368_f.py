# YOUR CODE HERE
import sys
import threading


def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N_MAX = 100005
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    divisors = [[] for _ in range(N_MAX)]
    for d in range(1, N_MAX):
        for n in range(2*d, N_MAX, d):
            divisors[n].append(d)
    GR = [0] * N_MAX
    for n in range(2, N_MAX):
        options = set()
        for d in divisors[n]:
            options.add(GR[d])
        mex = 0
        while mex in options:
            mex += 1
        GR[n] = mex
    nimber = 0
    for a in A:
        nimber ^= GR[a]
    if nimber == 0:
        print("Bruno")
    else:
        print("Anna")

threading.Thread(target=main).start()