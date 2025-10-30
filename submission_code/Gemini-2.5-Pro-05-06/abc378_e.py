import sys

# Fenwick tree (BIT) implementation
# It is 1-indexed internally. Operations take 0-indexed `val`
# and convert to 1-indexed for tree array.
class FenwickTree:
    def __init__(self, max_val_exclusive):
        # max_val_exclusive is M, for values 0 to M-1.
        # Max 1-based index used in tree array is M. Tree array size M+1.
        self.tree_size = max_val_exclusive 
        self.tree = [0] * (self.tree_size + 1)

    def update(self, val, delta):
        # val is 0-based (0 to M-1)
        # Convert to 1-based for BIT
        idx = val + 1
        while idx <= self.tree_size:
            self.tree[idx] += delta
            idx += idx & (-idx)

    def query_prefix_sum(self, val):
        # val is 0-based (0 to M-1)
        # Sums counts/values for original values from 0 up to val (inclusive)
        # If val is -1 (e.g. from val_low - 1 when val_low is 0), should return 0.
        if val < 0:
            return 0
        # Convert to 1-based for BIT
        idx = val + 1
        s = 0
        while idx > 0:
            s += self.tree[idx]
            idx -= idx & (-idx)
        return s
    
    def query_range_sum(self, val_low, val_high):
        # Calculates sum for values in [val_low, val_high] (0-indexed values)
        if val_low > val_high:
            return 0
        
        sum_high = self.query_prefix_sum(val_high)
        sum_low_minus_1 = self.query_prefix_sum(val_low - 1)
        return sum_high - sum_low_minus_1

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split()))

    total_ans = 0
    
    # Stores counts of P'_j encountered so far, indexed by their value
    fenwick_counts = FenwickTree(M) # For values 0 to M-1. Max index M.

    # Sum of P'_j encountered so far (P'_j = P_j mod M)
    sum_of_all_P_primes_so_far = 0
    
    # Count of P'_j encountered so far.
    count_of_P_primes_so_far = 0

    # Represents P_(-1) = 0 (prefix sum before any elements of A_list). So P'_{-1} = 0.
    current_P_val_actual = 0 
    current_P_prime_val = 0 # P'_{-1} = 0 % M = 0

    # Add P'_{-1} to data structures
    fenwick_counts.update(current_P_prime_val, 1)
    sum_of_all_P_primes_so_far += current_P_prime_val
    count_of_P_primes_so_far += 1

    for i in range(N): # Loop for A_list[0], ..., A_list[N-1]
        # current_P_val_actual is P_{i-1} (actual sum)
        # current_P_prime_val is P'_{i-1} (sum mod M)
        
        current_P_val_actual = current_P_val_actual + A_list[i]
        # This is P'_i (0-indexed based on A_list)
        current_P_prime_val = current_P_val_actual % M
        
        # The sum for this P'_i is Sum_{P'_j seen so far} ( (P'_i - P'_j + M) % M )
        # This equals:
        #   count_of_P_primes_so_far * P'_i 
        # - sum_of_all_P_primes_so_far
        # + (count of P'_j such that P'_j > P'_i among those seen so far) * M
        
        # N2_count = count of P'_j (among those seen so far) such that P'_j > current_P_prime_val
        # These P'_j have values in the range [current_P_prime_val + 1, M-1]
        N2_count = fenwick_counts.query_range_sum(current_P_prime_val + 1, M - 1)
        
        term_for_this_P_i_prime = (count_of_P_primes_so_far * current_P_prime_val - 
                                   sum_of_all_P_primes_so_far + 
                                   N2_count * M)
        total_ans += term_for_this_P_i_prime
        
        # Add current P'_i to data structures for next iterations
        fenwick_counts.update(current_P_prime_val, 1)
        sum_of_all_P_primes_so_far += current_P_prime_val
        count_of_P_primes_so_far += 1
        
    sys.stdout.write(str(total_ans) + "
")

solve()