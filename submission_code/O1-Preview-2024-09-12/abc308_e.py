# YOUR CODE HERE

import sys
import threading

def main():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    S = N_and_rest[N+1]
    
    N = len(A)
    countsM_prev = [0,0,0]  # counts of 'M's with A_i == 0,1,2
    countsME = [[0]*3 for _ in range(3)]  # countsME[A_e][A_m]
    ans = 0
    
    # Precompute mex[a][b][c] for all a,b,c ∈ {0,1,2}
    mex = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                s = set([a,b,c])
                for m in range(4):
                    if m not in s:
                        mex[a][b][c] = m
                        break
                else:
                    mex[a][b][c] = 3  # When s = {0,1,2}
    for i in range(N):
        a_i = A[i]
        s_i = S[i]
        if s_i == 'M':
            countsM_prev[a_i] +=1
        elif s_i == 'E':
            for A_m in range(3):
                countsME[a_i][A_m] += countsM_prev[A_m]
        elif s_i == 'X':
            for A_e in range(3):
                for A_m in range(3):
                    counts = countsME[A_e][A_m]
                    if counts > 0:
                        mex_val = mex[A_m][A_e][a_i]
                        ans += counts * mex_val
    print(ans)

threading.Thread(target=main).start()