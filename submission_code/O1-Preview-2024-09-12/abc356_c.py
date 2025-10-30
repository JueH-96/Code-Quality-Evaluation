# YOUR CODE HERE
import sys

import threading
def main():
    import sys
    import math
    input = sys.stdin.readline
    N, M, K = map(int, sys.stdin.readline().split())
    tests = []
    for _ in range(M):
        parts = sys.stdin.readline().split()
        Ci = int(parts[0])
        Ai = list(map(int, parts[1:1+Ci]))
        Ri = parts[-1]
        tests.append((Ai, Ri))

    ans = 0
    for mask in range(1 << N):
        ok = True
        for Ai, Ri in tests:
            num_real = 0
            for key in Ai:
                # Keys are numbered from 1 to N, bits from 0 to N-1
                if (mask >> (key -1)) &1:
                    num_real +=1
            if Ri == 'o':
                if num_real < K:
                    ok = False
                    break
            elif Ri == 'x':
                if num_real >= K:
                    ok = False
                    break
        if ok:
            ans +=1
    print(ans)
threading.Thread(target=main).start()