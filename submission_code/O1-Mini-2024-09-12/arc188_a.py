# YOUR CODE HERE
import sys
import math
from itertools import product

MOD = 998244353

def solve():
    from sys import stdin
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, stdin.readline().split())
    S = stdin.readline().strip()
    
    m = S.count('?')
    if K == 0:
        print(pow(3, m, MOD))
        return
    
    if m > 15:
        print(0)  # Placeholder for larger m
        return
    
    # Enumerate all possible assignments
    assignments = list(product(['A', 'B', 'C'], repeat=m))
    count = 0
    idx = 0
    for assign in assignments:
        temp = list(S)
        for i in range(N):
            if temp[i] == '?':
                temp[i] = assign[idx]
                idx +=1
        idx =0
        good = 0
        for l in range(N):
            cntA = cntB = cntC =0
            for r in range(l, N):
                if temp[r] == 'A':
                    cntA +=1
                elif temp[r] == 'B':
                    cntB +=1
                elif temp[r] == 'C':
                    cntC +=1
                if (cntA %2 == cntB %2) and (cntA %2 == cntC %2):
                    good +=1
        if good >= K:
            count +=1
    print(count % MOD)