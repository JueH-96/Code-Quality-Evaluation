def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    K = int(input_data[1])
    P = list(map(int, input_data[2:]))

    MOD = 998244353

    # ----------------------------------------------------------------
    # 1) We need a fast way to compute the total number of inversions in P.
    #    This is a standard Fenwick–tree (BIT) based approach in O(N log N).
    #    (Also known as "merge sort counting" or "BIT inversion count".)
    #
    #    get_total_inversions(P) will return the total number of inversions of P
    #    modulo a large integer, but since N can be up to 2e5, the inversion count
    #    can be up to ~ (N*(N-1))/2 ~ 2e10, which still fits into 64-bit integer.
    #    We'll do the final mod at the end.
    # ----------------------------------------------------------------

    class Fenwick:
        """Fenwick (BIT) for frequencies up to size = n (1-based)."""
        def __init__(self, n):
            self.n = n
            self.data = [0]*(n+1)

        def update(self, idx, val):
            # add val at index idx (1-based Fenwick)
            while idx <= self.n:
                self.data[idx] += val
                idx += (idx & -idx)

        def query(self, idx):
            # sum from 1..idx
            s = 0
            while idx > 0:
                s += self.data[idx]
                idx -= (idx & -idx)
            return s

    def get_total_inversions(arr):
        # arr is 0-based, but values range in 1..N (since P is a permutation).
        # We'll do a Fenwick of size N.
        fenw = Fenwick(N)
        inv_count = 0
        # Process from left to right
        for i, x in enumerate(arr):
            # x in [1..N]
            # how many have we seen so far that are > x?
            # number of elements processed so far = i
            # number of elements <= x is fenw.query(x)
            smaller_or_equal = fenw.query(x)
            bigger = i - smaller_or_equal
            inv_count += bigger
            fenw.update(x, 1)
        return inv_count

    # ----------------------------------------------------------------
    # 2) We need subInv(i): the number of inversions in the subarray P[i..i+K-1],
    #    for i = 0..(N-K).  (0-based indexing in code)
    #
    #    We'll slide a window of length K from left to right.  Maintain the
    #    inversion count for that window in O(log N) per step via another Fenwick.
    #
    #    - When we add a new element x to the RIGHT end of the window:
    #         new_inversions_formed = number_of_window_elements_that_are > x
    #    - When we remove an old element y from the LEFT of the window:
    #         old_inversions_removed = number_of_window_elements_that_are < y
    #       because those pairs (y, something_smaller) contributed to the total.
    #
    #    Actually, for inversion pair (r, s) with r < s, the new element is s,
    #    so the "new inversions" when we add x at the right are how many in the
    #    window are strictly greater than x.  That is count_of_window - fenw.query(x).
    #
    #    When removing y from the left, we remove all pairs (y, z) with z > y.
    #    But in the 0-based sense, y is at position r, any z is at position s>r,
    #    so z is added after y.  For inversion we need y>z.  So the pairs we remove
    #    is how many z are strictly less than y.  That is fenw.query(y-1).
    #
    #    Implementation detail is somewhat subtle, but this procedure is a known
    #    “sliding-window inversion count” trick.  We just have to be consistent
    #    with the code.
    # ----------------------------------------------------------------

    class FenwickWindow:
        """Maintain a running Fenwick for a sliding window of size K."""
        def __init__(self, n):
            self.n = n
            self.fenw = [0]*(n+1)
            self.size = 0  # how many currently in the window
            self.curr_inv = 0

        def fenw_update(self, idx, val):
            while idx <= self.n:
                self.fenw[idx] += val
                idx += (idx & -idx)

        def fenw_query(self, idx):
            s = 0
            while idx > 0:
                s += self.fenw[idx]
                idx -= (idx & -idx)
            return s

        def add_right(self, x):
            # x in [1..n]
            smaller_or_equal = self.fenw_query(x)
            # bigger than x = self.size - smaller_or_equal
            bigger = self.size - smaller_or_equal
            self.curr_inv += bigger
            self.fenw_update(x, 1)
            self.size += 1

        def remove_left(self, x):
            # remove the contribution of x from curr_inv
            # we remove those pairs (x, z) with x>z.  That is how many z < x
            smaller = self.fenw_query(x-1)
            self.curr_inv -= smaller
            self.fenw_update(x, -1)
            self.size -= 1

    # Compute all subInv(i) for i=0..N-K
    # We'll store them in an array sub_inv[], length = N-K+1
    def get_sub_inversions(arr, K):
        if K <= 1:
            # subarray of length 1 has 0 inversions
            return [0]*(N-K+1)

        fw = FenwickWindow(N)
        # build first window
        for j in range(K):
            fw.add_right(arr[j])

        sub_inv = [0]*(N-K+1)
        sub_inv[0] = fw.curr_inv

        for i in range(1, N-K+1):
            # remove arr[i-1], add arr[i+K-1]
            fw.remove_left(arr[i-1])
            fw.add_right(arr[i+K-1])
            sub_inv[i] = fw.curr_inv

        return sub_inv

    # ----------------------------------------------------------------
    # 3) According to a well-known (and also derivable) identity for this problem:
    #
    #    Expected number of inversions after:
    #       =  #Inv(P)  +  (K*(K-1))/4   –  average( subInv(i) )
    #
    #    where subInv(i) is the inversion-count of the length-K block at position i.
    #
    #    Explanation (short version): 
    #    - The block [i..i+K-1] itself goes from subInv(i) inversions to an average of K(K-1)/4
    #      when uniformly permuted.  
    #    - The cross-boundary pairs remain the same "on average."  
    #    Thus overall difference is [K(K-1)/4 - subInv(i)] for that block.  
    #    Then we average over i uniformly, i=0..N-K => final formula above.
    # ----------------------------------------------------------------

    # 4) Implement all steps:

    # Step (a): total inversions of P
    invP = get_total_inversions(P)  # up to about 2e10, fits 64-bit

    # Step (b): subInv array
    if K == 1:
        # Then subInv(i)=0 for all i.  So average = 0.
        # Expected inversions = invP + 0 - 0 = invP.
        # We will still compute it via formula for uniformity, but it's trivial.
        sub_inv_array = [0]*(N-K+1)
    else:
        sub_inv_array = get_sub_inversions(P, K)

    sub_inv_sum = sum(sub_inv_array)

    # Step (c): K(K-1)/4 in modular form
    # We do everything in fraction form mod 998244353.  Let’s define:
    #   K*(K-1) // 2 is the number of inversions of a K-length sequence if it were sorted descending,
    #   but we want half of that again => K*(K-1)/4 => we multiply K*(K-1) by inv4 mod.
    #
    # We'll do:
    #   half = (K*(K-1)//2)
    #   expected_in_subblock = half/2 = K*(K-1)/4
    #
    # but to keep it purely in modular arithmetic (where dividing by 2 means multiplying by
    # the inverse of 2 mod 998244353, etc.), we do it systematically.

    # precompute inverses of 2 and 4.
    inv2 = pow(2, MOD-2, MOD)
    inv4 = pow(4, MOD-2, MOD)

    # For safety if K*(K-1) is big, do it in 64-bit then mod:
    KK = K*(K-1)  # up to ~4e10, still fits 64-bit
    KK_mod = KK % MOD

    # We want (K(K-1))/4 mod => do KK_mod * inv4 mod
    KK_div4 = (KK_mod * inv4) % MOD

    # Step (d): average of subInv => (sum_of_subInv / (N-K+1)) in mod
    n_k_1 = N - K + 1
    # inverse of (N-K+1) mod
    inv_n_k_1 = pow(n_k_1, MOD-2, MOD)

    # sub_inv_sum might be up to ~ K*(K-1)/2 times (N-K+1), can be up to ~2e10 * 2e5 = 4e15 (64-bit OK)
    sub_inv_sum_mod = sub_inv_sum % MOD
    avg_sub_inv_mod = (sub_inv_sum_mod * inv_n_k_1) % MOD

    # Step (e): final =  #Inv(P) + K(K-1)/4 - average_subInv  (all mod)
    invP_mod = invP % MOD

    # carefully do final = invP_mod + KK_div4 - avg_sub_inv_mod (mod)
    # handle negative by adding MOD if needed
    ans = invP_mod + KK_div4 - avg_sub_inv_mod
    ans %= MOD

    # ----------------------------------------------------------------
    # Print the result
    # ----------------------------------------------------------------
    print(ans)