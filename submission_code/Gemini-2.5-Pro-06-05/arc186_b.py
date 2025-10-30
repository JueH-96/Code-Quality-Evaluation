import sys

# It is recommended to use iterative version of find for deep recursion.
sys.setrecursionlimit(3 * 10**5 + 5)

def main():
    """
    Main function to solve the problem.
    """
    try:
        input = sys.stdin.readline
        N = int(input())
        if N == 0:
            print(1)
            return
        A_input = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        print(1)
        return

    # Adjust A to be 1-indexed
    A = [0] * (N + 1)
    for i in range(N):
        A[i + 1] = A_input[i]

    MOD = 998244353

    # Precompute factorials and their modular inverses for combinations
    MAX_N = N + 5
    fact = [1] * MAX_N
    inv_fact = [1] * MAX_N
    for i in range(1, MAX_N):
        fact[i] = (fact[i - 1] * i) % MOD

    inv_fact[MAX_N - 1] = pow(fact[MAX_N - 1], MOD - 2, MOD)
    for i in range(MAX_N - 2, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n - r]) % MOD
        return (num * den) % MOD

    # Initialize DSU data structures
    parent = list(range(N + 2))
    size = [1] * (N + 2)
    ways = [1] * (N + 2)
    min_idx = list(range(N + 2))

    # DSU find operation with path compression
    def find(i):
        if parent[i] == i:
            return i
        root = find(parent[i])
        parent[i] = root
        return root

    for i in range(1, N + 1):
        # Phase 1: Merge groups for j in (A_i, i) with i, where P_i is the minimum.
        
        # Collect distinct roots for j in (A_i, i) efficiently using min_idx jump
        roots_to_merge = []
        j = i - 1
        while j > A[i]:
            r = find(j)
            roots_to_merge.append(r)
            j = min_idx[r] - 1
        
        # Merge collected groups with i's group
        root_i = find(i)
        current_size = size[root_i]
        current_ways = ways[root_i]

        for r in roots_to_merge:
            # Combine group of i and group r.
            # P_i is the new minimum. The non-minimum elements are interleaved.
            # Number of ways to interleave B into A is C(|A|+|B|, |B|).
            # Here, we interleave size[r] elements with current_size-1 elements.
            current_ways = (current_ways * ways[r]) % MOD
            current_ways = (current_ways * nCr_mod(current_size - 1 + size[r], size[r])) % MOD
            
            # Union: merge group r into group root_i
            parent[r] = root_i
            current_size += size[r]
            min_idx[root_i] = min(min_idx[root_i], min_idx[r])
        
        size[root_i] = current_size
        ways[root_i] = current_ways
        
        # Phase 2: Merge with parent A_i group based on P_{A_i} < P_i
        if A[i] > 0:
            root_Ai = find(A[i])
            root_i = find(i)
            
            # The guarantee ensures root_Ai != root_i
            s_i = size[root_i]
            w_i = ways[root_i]
            s_Ai = size[root_Ai]
            w_Ai = ways[root_Ai]
            
            # P_i is min of its group. P_{A_i} < P_i means we interleave
            # elements of group i with elements of group A_i (excluding A_i itself).
            new_ways = (w_i * w_Ai) % MOD
            new_ways = (new_ways * nCr_mod(s_i + s_Ai - 1, s_i)) % MOD
            
            # Union: merge group i into group Ai
            parent[root_i] = root_Ai
            size[root_Ai] += s_i
            min_idx[root_Ai] = min(min_idx[root_Ai], min_idx[root_i])
            ways[root_Ai] = new_ways

    # All indices end up in one group. Find the root and print the ways.
    final_root = find(1)
    print(ways[final_root])

if __name__ == "__main__":
    main()