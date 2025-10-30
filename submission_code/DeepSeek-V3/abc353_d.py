# YOUR CODE HERE
import sys

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute the lengths of all A_i
    lengths = [len(str(a)) for a in A]
    
    # Precompute the powers of 10 up to the maximum length
    max_len = max(lengths)
    power = [1] * (max_len + 1)
    for i in range(1, max_len + 1):
        power[i] = (power[i-1] * 10) % MOD
    
    # Create a frequency map for each length
    from collections import defaultdict
    len_count = defaultdict(int)
    for l in lengths:
        len_count[l] += 1
    
    # Precompute the sum of A_i for each length
    len_sum = defaultdict(int)
    for a, l in zip(A, lengths):
        len_sum[l] = (len_sum[l] + a) % MOD
    
    # Calculate the total sum
    total = 0
    for i in range(N):
        a = A[i]
        l_a = lengths[i]
        for l_b, cnt in len_count.items():
            # Calculate the contribution when A_i is first
            # f(a, b) = a * 10^l_b + b
            # Sum over all b with length l_b
            # sum_{b} (a * 10^l_b + b) = a * 10^l_b * cnt + sum_{b} b
            # sum_{b} b is len_sum[l_b]
            term1 = (a * power[l_b]) % MOD
            term1 = (term1 * cnt) % MOD
            term2 = len_sum[l_b]
            total = (total + term1 + term2) % MOD
    
    # Since each pair (i,j) with i < j is counted once, but in the above loop, each pair is counted twice (once as (i,j) and once as (j,i)), we need to subtract the cases where i == j
    # However, since i < j, we don't have to subtract anything
    
    # But in the above loop, for each i, we consider all j, including j < i and j == i
    # So we need to subtract the cases where j < i and j == i
    # The total sum includes all pairs (i,j) where i != j, and each pair is counted once
    
    # Wait, no. The above loop counts each pair (i,j) where i < j once, and (j,i) once, but since we are summing over all i and all j, and then dividing by 2, but since we are not, we need to adjust
    
    # Alternatively, we can think of the total sum as the sum over all i and j of f(A_i, A_j), but since f(A_i, A_j) is not equal to f(A_j, A_i), we need to consider all pairs (i,j) where i != j
    
    # But in the problem, we are to sum over i < j, so we need to adjust the total sum
    
    # The above loop counts each pair (i,j) where i < j once, and (j,i) once, but since f(A_i, A_j) is not equal to f(A_j, A_i), we need to consider all pairs (i,j) where i != j
    
    # So the total sum is the sum over all i and j of f(A_i, A_j), but we need to subtract the cases where i == j
    
    # So the final answer is (total - sum_{i} f(A_i, A_i)) / 2
    
    # Compute sum_{i} f(A_i, A_i)
    sum_f_ii = 0
    for i in range(N):
        a = A[i]
        l_a = lengths[i]
        f_ii = (a * power[l_a] + a) % MOD
        sum_f_ii = (sum_f_ii + f_ii) % MOD
    
    # The total sum is the sum over all i and j of f(A_i, A_j), but we need to subtract the cases where i == j
    # So the sum over i < j is (total - sum_f_ii) / 2
    # But since we are working modulo MOD, we need to multiply by the modular inverse of 2
    inv2 = pow(2, MOD-2, MOD)
    result = (total - sum_f_ii) * inv2 % MOD
    print(result)

if __name__ == "__main__":
    main()