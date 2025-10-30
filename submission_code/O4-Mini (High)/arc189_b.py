#!/usr/bin/env python3
import sys
import threading

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    X = [int(next(it)) for _ in range(N)]
    # Build difference lists by parity of index j (1-based)
    # d[j] = X[j-1] - X[j-2] for j=2..N
    # If j even -> goes to E_list, if j odd -> O_list
    E = []
    O = []
    # idx runs from 1 to N-1, j = idx+1
    for idx in range(1, N):
        diff = X[idx] - X[idx-1]
        j = idx + 1
        if (j & 1) == 0:
            E.append(diff)
        else:
            O.append(diff)
    E.sort()
    O.sort()
    # Compute minimal sum:
    # sum p[i] = N*X[1] + sum_{j=2..N} (N+1-j)*d_new[j]
    # We assign sorted E to even j=2,4,... and sorted O to odd j=3,5,...
    total = N * X[0]
    # even diffs
    # E[k] -> j = 2*(k+1), weight = N+1 - j
    for k, val in enumerate(E):
        j = 2*(k+1)
        w = (N + 1) - j
        total += w * val
    # odd diffs
    # O[k] -> j = 2*(k+1)+1, weight = N+1 - j
    for k, val in enumerate(O):
        j = 2*(k+1) + 1
        w = (N + 1) - j
        total += w * val

    print(total)

if __name__ == "__main__":
    main()