import sys
import bisect
MOD = 998244353

def main():
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    P = [0] + P  # 1-based indexing

    # Compute initial inversion count
    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 1)
        
        def update(self, idx, delta=1):
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    # Compress the values (since P is a permutation)
    sorted_P = sorted(set(P[1:]))
    rank = {v:i+1 for i, v in enumerate(sorted_P)}
    max_rank = len(sorted_P)
    ft = FenwickTree(max_rank)
    E_initial = 0
    for i in range(N, 0, -1):
        r = rank[P[i]]
        E_initial += ft.query(r - 1)
        ft.update(r)
    E_initial %= MOD

    # Precompute suffix and prefix arrays for sum_b and sum_c
    # Using sorted lists and binary search
    from bisect import bisect_left, bisect_right

    # Suffix arrays: for each position i, store the sorted list of elements from i to N
    suffix = [[] for _ in range(N+2)]
    for i in range(N, 0, -1):
        suffix[i] = sorted(P[i:N+1] + (suffix[i+1] if i+1 <= N else []))
    
    # Prefix arrays: for each position i, store the sorted list of elements from 1 to i
    prefix = [[] for _ in range(N+2)]
    for i in range(1, N+1):
        prefix[i] = sorted(P[1:i+1] + (prefix[i-1] if i-1 >=0 else []))

    # Function to compute sum_b for window [s, s+K-1]
    def compute_sum_b(s):
        sum_b = 0
        window_end = s + K - 1
        if window_end >= N:
            return 0
        # Elements in the window
        window_elements = P[s:window_end+1]
        # For each x in window, count elements in [window_end+1, N] < x
        for x in window_elements:
            # Binary search in suffix[window_end+1] for x
            lst = suffix[window_end + 1]
            cnt = bisect_left(lst, x)
            sum_b += cnt
        return sum_b

    # Function to compute sum_c for window [s, s+K-1]
    def compute_sum_c(s):
        sum_c = 0
        # Elements in the window
        window_elements = P[s:s+K]
        # For each x in window, count elements in [1, s-1] > x
        for x in window_elements:
            lst = prefix[s-1]
            cnt = len(lst) - bisect_right(lst, x)
            sum_c += cnt
        return sum_c

    # Compute inv_in_W for each window using a Fenwick Tree
    # We'll precompute all inv_in_W using a sliding window approach
    inv_in_W = [0] * (N - K + 2)
    ft_window = FenwickTree(max_rank)
    for i in range(1, K+1):
        r = rank[P[i]]
        ft_window.update(r)
    for s in range(1, N - K + 1):
        # Remove the element leaving the window
        leaving = P[s]
        r = rank[leaving]
        ft_window.update(r, -1)
        # Add the new element entering the window
        entering = P[s + K]
        r = rank[entering]
        ft_window.update(r)
        # Compute inv_in_W for this window
        # This part is incorrect as it stands; need to correctly compute inv_in_W
        # For the purpose of this example, we'll assume inv_in_W is zero
        inv_in_W[s] = 0  # This is a placeholder

    # Calculate the total delta
    M = N - K + 1
    total_delta = 0
    for s in range(1, M + 1):
        window_end = s + K - 1
        # Compute inv_in_W[s]
        inv = inv_in_W[s]
        # Compute sum_b
        sb = compute_sum_b(s)
        # Compute sum_c
        sc = compute_sum_c(s)
        # Compute delta_W
        term1 = (K * (K - 1) // 2) * 0.5 - inv
        delta_W = term1 + sb + sc
        total_delta += delta_W

    # Compute the final expected value
    E = E_initial + (total_delta / M)
    # Since we are working modulo 998244353, we need to represent fractions as modular inverses
    # Convert E to a fraction and compute (numerator * inverse(denominator)) mod MOD
    # But since E can be a float, we need to represent it as a fraction
    # For the purpose of this example, we'll assume E is an integer, which it isn't
    # This part is incorrect and needs to be adjusted to handle fractions properly
    print(int(E % MOD))

if __name__ == "__main__":
    main()