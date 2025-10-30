#!/usr/bin/env python3
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    
    # Check possibility if any single dish triggers a stop (remember trigger is > threshold)
    ans = float('inf')
    for i in range(N):
        if A[i] > X or B[i] > Y:
            ans = 1
            break

    # Candidate based on sweetness threshold:
    candidateA = float('inf')
    # sort sweetness in descending order
    sortedA = sorted(A, reverse=True)
    s = 0
    for i in range(N):
        s += sortedA[i]
        if s > X:
            # In this prefix, the candidate 'last dish' is sortedA[0], the biggest dish.
            # If the sum of the remaining (s - sortedA[0]) is <= X then ordering is possible.
            if s - sortedA[0] <= X:
                candidateA = i + 1
            break

    # Candidate based on saltiness threshold:
    candidateB = float('inf')
    sortedB = sorted(B, reverse=True)
    s = 0
    for i in range(N):
        s += sortedB[i]
        if s > Y:
            if s - sortedB[0] <= Y:
                candidateB = i + 1
            break

    candidate = min(candidateA, candidateB)
    if candidate == float('inf'):
        candidate = N  # fallback (with all dishes the stop is guaranteed)
    
    # Also consider the case when a one-dish solution existed.
    ans = min(ans, candidate)
    sys.stdout.write(str(ans) + "
")
    
if __name__ == '__main__':
    main()