# YOUR CODE HERE
import sys, math
import numpy as np

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    A = np.array(list(map(int, data[2:])), dtype=np.int64)
    N_int = N
    # Extend the array to simulate two cycles.
    A_ext = np.concatenate([A, A])
    # Build cumulative sum array P (length = 2N+1)
    P = np.concatenate(([0], np.cumsum(A_ext)))
    size = 2 * N_int + 1
    INF = 10**9  # A number larger than any possible index (max index <= 400k)

    # Given candidate x, we want to decide quickly whether there exists a rotation r (0<=r<N)
    # so that the cake rotated with first piece A[r] (i.e. pieces r, r+1, ..., r+N-1) can be partitioned into K segments 
    # using the greedy rule (i.e. "jump" procedure). 
    # We define a function feasible(x) that returns True if some rotation is valid.
    def feasible(x):
        # Compute jump pointer f for each index i in 0..2N:
        queries = P + x
        f_arr = np.searchsorted(P, queries, side='left').astype(np.int64)
        f_arr = np.where(f_arr < size, f_arr, INF)

        # Build jump table J with Lmax levels; J[0] = f_arr.
        Lmax = int(math.ceil(math.log2(K+1))) if K > 0 else 1
        J = [None] * Lmax
        J[0] = f_arr
        for lvl in range(1, Lmax):
            prev = J[lvl-1]
            curr = np.empty_like(prev)
            valid_mask = (prev < size)
            # For indices where valid_mask is True, set curr[i] = J[lvl-1][ prev[i] ]
            curr[valid_mask] = J[lvl-1][prev[valid_mask]]
            curr[~valid_mask] = INF
            J[lvl] = curr
        # For each rotation r in 0..N-1, compute jump(r, K)
        indices = np.arange(N_int, dtype=np.int64)
        cur = indices.copy()
        rem = K
        bit = 0
        while rem:
            if rem & 1:
                cur = J[bit][cur]
            rem //= 2
            bit += 1
        # A rotated cake (starting at r) is feasible if jump(r, K) <= r+N.
        valid = (cur <= indices + N_int)
        return valid.any()
    
    # Binary search for the maximum candidate x (1 <= x <= total//K)
    total = int(A.sum())
    lo = 1
    hi = (total // K) + 1
    x_opt = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            x_opt = mid
            lo = mid + 1
        else:
            hi = mid

    # Now recompute the boolean array valid_full for each rotation,
    # for x_opt.
    def compute_valid_array(x):
        queries = P + x
        f_arr = np.searchsorted(P, queries, side='left').astype(np.int64)
        f_arr = np.where(f_arr < size, f_arr, INF)
        Lmax = int(math.ceil(math.log2(K+1))) if K > 0 else 1
        J = [None] * Lmax
        J[0] = f_arr
        for lvl in range(1, Lmax):
            prev = J[lvl-1]
            curr = np.empty_like(prev)
            valid_mask = (prev < size)
            curr[valid_mask] = J[lvl-1][prev[valid_mask]]
            curr[~valid_mask] = INF
            J[lvl] = curr
        indices = np.arange(N_int, dtype=np.int64)
        cur = indices.copy()
        rem = K
        bit = 0
        while rem:
            if rem & 1:
                cur = J[bit][cur]
            rem //= 2
            bit += 1
        valid_full = (cur <= indices + N_int)
        return valid_full

    valid_full = compute_valid_array(x_opt)
    # By rotational symmetry, a cut line i (0-indexed, meaning it lies between piece i and piece (i+1))
    # is cut in some optimal partition if and only if the rotation that makes piece (i+1) first (i.e. index (i+1)%N)
    # is feasible.
    # Thus form a cyclic shift by one: For each i, the indicator for the cut line i
    # is valid_full[(i+1) % N]. Then, the number of cut lines that are cut in some optimal partition is the count of True's.
    shifted = np.roll(valid_full, -1)   # shifted so that shifted[i] = valid_full[(i+1)%N]
    num_cut = int(shifted.sum())
    y = N_int - num_cut

    sys.stdout.write(f"{x_opt} {y}")

if __name__ == '__main__':
    main()