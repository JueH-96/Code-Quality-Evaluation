import sys, sys
import numpy as np
from numba import njit, prange

#--------------------------------------------------------------------
# Given a sorted subarray S = A[L:R+1] (with L,R as 0-indexed endpoints)
# the function greedy_interval returns the maximum number of kagamimochi
# (i.e. pairs) that can be formed – using the greedy two–pointer algorithm.
#
# In a sorted subarray S (of length n), one can only possibly form at most floor(n/2)
# pairs. The algorithm “pretends” that the first floor(n/2) elements of S are the candidate
# “tops” (to be placed on top) and then uses a pointer through the remaining part of S to find
# a “bottom” for each top so that 2*S[i] <= S[j]. 
@njit
def greedy_interval(A, L, R):
    n = R - L + 1
    half = n >> 1  # floor(n/2)
    i = 0
    j = half
    count = 0
    # We treat the query subarray as S = A[L ... R].
    # i runs from 0 to half-1 (i.e. corresponds to indices L ... L+half-1)
    # j runs over the rest (indices L+half ... R)
    while i < half and j < n:
        # S[i] is A[L+i] and S[j] is A[L+j]
        if 2 * A[L + i] <= A[L + j]:
            count += 1
            i += 1
            j += 1
        else:
            j += 1
    return count

#--------------------------------------------------------------------
# process_queries takes in the sorted array A, the number Q of queries,
# and a Qx2 integer array qs (each row is [L,R] with 1-indexed positions).
# It returns an array "res" with the answer for each query computed using
# the greedy_interval function.
@njit(parallel=True)
def process_queries(A, Q, qs):
    res = np.empty(Q, dtype=np.int64)
    for q in prange(Q):
        L = qs[q, 0]
        R = qs[q, 1]
        # Convert to 0-indexed.
        start = L - 1
        end = R - 1
        res[q] = greedy_interval(A, start, end)
    return res

#--------------------------------------------------------------------
def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    # Read the N mochi sizes (they are sorted in non-decreasing order).
    A = np.empty(N, np.int64)
    for i in range(N):
        A[i] = int(next(it))
    Q = int(next(it))
    qs = np.empty((Q, 2), np.int64)
    for i in range(Q):
        L = int(next(it))
        R = int(next(it))
        qs[i, 0] = L
        qs[i, 1] = R
    # Process all queries (using numba parallel loop).
    out = process_queries(A, Q, qs)
    # Print one answer per line.
    out_lines = "
".join(str(x) for x in out)
    sys.stdout.write(out_lines)

if __name__=="__main__":
    main()