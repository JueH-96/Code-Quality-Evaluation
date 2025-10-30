MOD = 998244353

def inv(x, mod):
    return pow(x, mod-2, mod)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:2+N]))
    
    # Precompute factorials and inverse factorials
    fact = [1] * (K+1)
    for i in range(1, K+1):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (K+1)
    inv_fact[K] = inv(fact[K], MOD)
    for i in range(K-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    # Expected inversion number for a random permutation of size K
    expected_inv = K * (K-1) // 2 * inv(2, MOD) % MOD
    
    # Total number of possible operations
    total_ops = N - K + 1
    
    # Calculate the initial inversion count
    inv_count = 0
    # Using a BIT to count inversions
    class BIT:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)
        
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
    
    bit = BIT(N)
    for num in reversed(P):
        inv_count += bit.query(num-1)
        bit.update(num)
    
    # Calculate the expected inversion count after operations
    # The expected inversion count is:
    # initial_inv - sum_{i=1}^{N-K+1} (inv_count of the K-length segment) / total_ops
    # + sum_{i=1}^{N-K+1} expected_inv / total_ops
    
    # To compute the sum of inversion counts of all K-length segments
    # We need to find the inversion count for each segment
    # But it's computationally expensive to do it directly
    # Instead, we can use the fact that the inversion count of a segment is the same as the inversion count of the permutation of that segment
    
    # However, since the segments overlap, we need a smarter way
    # We can precompute the inversion counts for all segments using a sliding window approach
    
    # But given the constraints, it's not feasible to compute for each segment
    # So, we need a different approach
    
    # Let's consider the contribution of each pair (i,j) to the inversion count
    # For each pair (i,j) where i < j, if P[i] > P[j], it contributes 1 to the inversion count
    # After the operation, if the segment [i, i+K-1] or [j, j+K-1] contains both i and j, the contribution may change
    
    # So, for each pair (i,j), we need to determine the probability that the segment chosen affects the pair
    
    # The probability that a segment contains both i and j is:
    # The number of segments that contain both i and j divided by total_ops
    
    # The number of segments that contain both i and j is:
    # max(0, K - (j - i)) if i < j
    
    # So, for each pair (i,j) where i < j and P[i] > P[j], the probability that the segment affects the pair is:
    # (K - (j - i)) / total_ops if j - i < K, else 0
    
    # If the segment affects the pair, the expected inversion count for the pair is expected_inv / (K * (K-1) / 2)
    # Because the expected inversion count for a random permutation of size K is K*(K-1)/4
    
    # So, the expected inversion count after the operation is:
    # initial_inv - sum_{i<j, P[i] > P[j]} (1 * (K - (j - i)) / total_ops) + sum_{i<j, P[i] > P[j} (expected_inv / (K*(K-1)/2) * (K - (j - i)) / total_ops)
    
    # Wait, perhaps it's easier to compute the expected inversion count directly
    
    # Let's think differently: the expected inversion count after the operation is the initial inversion count minus the expected number of inversions that are destroyed by the operation, plus the expected number of new inversions created by the operation
    
    # The expected number of inversions destroyed is the sum over all pairs (i,j) where i < j and P[i] > P[j] of the probability that the segment chosen contains both i and j
    
    # The expected number of new inversions created is the expected number of inversions in the shuffled segment
    
    # So, the expected inversion count after the operation is:
    # initial_inv - sum_{i<j, P[i] > P[j]} (K - (j - i)) / total_ops + expected_inv * (number of segments) / total_ops
    
    # Wait, no. The expected number of new inversions is expected_inv * (number of segments) / total_ops, but the number of segments is total_ops
    
    # So, the expected inversion count after the operation is:
    # initial_inv - sum_{i<j, P[i] > P[j]} (K - (j - i)) / total_ops + expected_inv
    
    # Now, we need to compute the sum of (K - (j - i)) for all pairs (i,j) where i < j and P[i] > P[j]
    
    # To compute this efficiently, we can use a BIT to count the number of elements less than P[i] to the right of i, and multiply by (K - (j - i))
    
    # But it's not straightforward. Instead, we can precompute for each i, the number of j > i where P[j] < P[i], and then for each such j, compute (K - (j - i))
    
    # However, this is still O(N^2) in the worst case, which is not acceptable
    
    # So, we need a smarter way
    
    # Let's consider the contribution of each element P[i] to the sum
    # For each i, the number of j > i where P[j] < P[i] is the number of inversions involving P[i]
    # For each such j, the contribution is (K - (j - i))
    
    # So, for each i, we need to find the sum of (K - (j - i)) for all j > i where P[j] < P[i]
    
    # This can be rewritten as K * count - sum_{j > i, P[j] < P[i]} (j - i)
    
    # So, for each i, we need to compute:
    # count: the number of j > i where P[j] < P[i]
    # sum_j: the sum of (j - i) for all j > i where P[j] < P[i]
    
    # Then, the total sum is sum_{i} (K * count_i - sum_j_i)
    
    # We can compute count_i and sum_j_i using a BIT
    
    # Initialize a BIT to count the number of elements less than P[i] to the right of i
    # And another BIT to sum the indices of elements less than P[i] to the right of i
    
    bit_count = BIT(N)
    bit_sum = BIT(N)
    
    total = 0
    for i in range(N-1, -1, -1):
        count = bit_count.query(P[i]-1)
        sum_j = bit_sum.query(P[i]-1)
        total += (K * count - (sum_j - i * count)) % MOD
        bit_count.update(P[i], 1)
        bit_sum.update(P[i], i+1)
    
    # Now, the expected inversion count after the operation is:
    # initial_inv - total / total_ops + expected_inv
    
    # Since we are working modulo MOD, we need to compute the modular inverse of total_ops
    inv_total_ops = inv(total_ops, MOD)
    
    expected = (inv_count - total * inv_total_ops % MOD + expected_inv) % MOD
    
    print(expected)

if __name__ == "__main__":
    main()