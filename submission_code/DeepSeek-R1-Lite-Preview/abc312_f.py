import sys
import bisect

def main():
    N, M = map(int, sys.stdin.readline().split())
    P = []
    R = []
    C = []
    for _ in range(N):
        T, X = map(int, sys.stdin.readline().split())
        if T == 0:
            P.append(X)
        elif T == 1:
            R.append(X)
        elif T == 2:
            C.append(X)
    
    # Sort in descending order
    P.sort(reverse=True)
    R.sort(reverse=True)
    C.sort(reverse=True)
    
    # Compute prefix sums
    prefix_P = [0]
    for x in P:
        prefix_P.append(prefix_P[-1] + x)
    
    prefix_R = [0]
    for x in R:
        prefix_R.append(prefix_R[-1] + x)
    
    prefix_C = [0]
    for x in C:
        prefix_C.append(prefix_C[-1] + x)
    
    max_happiness = 0
    len_C = len(C)
    len_R = len(R)
    len_P = len(P)
    
    # Iterate over possible number of regular cans to select
    for r in range(0, min(M, len_R) + 1):
        # Find the smallest k such that prefix_C[k] >= r
        k = bisect.bisect_left(prefix_C, r)
        if k > len_C:
            continue  # Not possible
        p = M - k - r
        if p >= 0 and p <= len_P:
            total_happiness = prefix_R[r] + (prefix_P[p] if p > 0 else 0)
            if total_happiness > max_happiness:
                max_happiness = total_happiness
    print(max_happiness)

if __name__ == '__main__':
    main()