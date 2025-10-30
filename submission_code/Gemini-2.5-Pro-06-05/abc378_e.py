import sys

def main():
    """
    Solves the problem by reformulating the sum and using a Fenwick tree
    for counting inversions.
    """

    class FenwickTree:
        """
        Fenwick Tree for prefix sum queries and point updates.
        Used here to count inversions efficiently.
        """
        def __init__(self, size):
            # The tree handles 1-based indices from 1 to size.
            self.tree = [0] * (size + 1)
            self.size = size

        def add(self, i, x):
            """Add x to index i (1-based)."""
            while i <= self.size:
                self.tree[i] += x
                i += i & -i

        def query(self, i):
            """Get sum of elements from 1 to i (inclusive, 1-based)."""
            s = 0
            while i > 0:
                s += self.tree[i]
                i -= i & -i
            return s

    # Read input using fast I/O
    fast_input = sys.stdin.readline
    try:
        N, M = map(int, fast_input().split())
        A = list(map(int, fast_input().split()))
    except (IOError, ValueError):
        # Handle empty input case for local testing
        return

    # S is the prefix sum array modulo M.
    # S[k] = (A_0 + ... + A_{k-1}) mod M for k > 0
    # S[0] is defined as 0.
    # The size of S is N+1.
    S = [0] * (N + 1)
    current_sum = 0
    for i in range(N):
        current_sum += A[i]
        S[i+1] = current_sum % M

    # The total sum is split into two parts: sum_term and M * inversions.
    
    # Part 1: Calculate sum_term = sum_{0<=j<r<=N} (S_r - S_j)
    # This is equivalent to sum_{r=1..N} r*S_r - sum_{j=0..N-1} (N-j)*S_j
    sum_term = 0
    for i in range(1, N + 1):
        sum_term += i * S[i]
    for j in range(N): # Loop from j=0 to N-1
        sum_term -= (N - j) * S[j]

    # Part 2: Calculate the number of inversions in S_0, ..., S_N
    inversions = 0
    # The Fenwick Tree stores frequencies of values seen so far.
    # Values in S are in [0, M-1]. We map value `v` to index `v+1` in the FT.
    ft = FenwickTree(M)

    for k in range(N + 1):
        val = S[k]
        # Number of elements processed before S[k] is `k` (from S[0] to S[k-1]).
        # Count how many of these are greater than S[k].
        # `ft.query(val + 1)` gives the count of elements seen so far that are <= `val`.
        num_less_equal = ft.query(val + 1)
        num_greater = k - num_less_equal
        inversions += num_greater
        
        # Add the current value's frequency to the Fenwick Tree.
        ft.add(val + 1, 1)

    # The final answer is the sum of the two parts.
    total_sum = sum_term + M * inversions
    print(total_sum)

if __name__ == "__main__":
    main()