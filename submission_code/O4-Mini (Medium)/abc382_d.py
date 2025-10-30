#!/usr/bin/env python3
import sys
sys.setrecursionlimit(1000000)

def main():
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])

    # Precompute the maximum allowed value for A[idx] so that
    # the remaining (N-1-idx) values can still increase by at least 10 each.
    # hi_bound[idx] = M - 10*(N-1-idx)
    hi_bound = [M - 10*(N-1-idx) for idx in range(N)]

    # We'll do two DFS passes:
    #  1) Count all valid sequences
    #  2) Print them in lexicographical order as we generate them

    arr = [0]*N
    total = 0

    # First pass: count
    def dfs_count(idx, prev):
        nonlocal total
        if idx == N:
            # We've filled all N positions successfully
            total += 1
            return
        # determine lower bound for this position
        lo = 1 if idx == 0 else prev + 10
        hi = hi_bound[idx]
        if lo > hi:
            return
        for v in range(lo, hi+1):
            arr[idx] = v
            dfs_count(idx+1, v)

    dfs_count(0, 0)

    # Output the count first
    out = sys.stdout
    out.write(str(total) + "
")

    # Second pass: generate & print
    def dfs_print(idx, prev):
        if idx == N:
            # print one complete sequence
            out.write(" ".join(map(str, arr)))
            out.write("
")
            return
        lo = 1 if idx == 0 else prev + 10
        hi = hi_bound[idx]
        if lo > hi:
            return
        for v in range(lo, hi+1):
            arr[idx] = v
            dfs_print(idx+1, v)

    dfs_print(0, 0)

if __name__ == "__main__":
    main()