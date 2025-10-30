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
    
    # Compute the sum of A_i multiplied by the appropriate power of 10
    # for each possible length
    sum_by_len = {}
    for a, l in zip(A, lengths):
        if l not in sum_by_len:
            sum_by_len[l] = 0
        sum_by_len[l] = (sum_by_len[l] + a) % MOD
    
    # Compute the total sum
    total = 0
    for i in range(N):
        a = A[i]
        l = lengths[i]
        # For each possible length of the second number
        for l2 in sum_by_len:
            # The contribution is a * 10^l2 * count of numbers with length l2
            # but only for j > i
            # So we need to subtract the cases where j <= i
            # But since we are iterating i from 0 to N-1, and j from i+1 to N
            # We can precompute the total sum and subtract the sum up to i
            # Alternatively, we can precompute the sum for all j and subtract the sum for j <= i
            # But it's easier to precompute the total sum and subtract the sum for j <= i
            # So we need to know the sum of A_j for j > i and length l2
            # We can precompute the prefix sums for each length
            # So we need to precompute the prefix sums for each length
            # Let's precompute the prefix sums for each length
            # Initialize prefix sums
            if not hasattr(main, 'prefix_sums'):
                main.prefix_sums = {}
                for l3 in sum_by_len:
                    main.prefix_sums[l3] = [0] * (N + 1)
                    for idx in range(N):
                        main.prefix_sums[l3][idx+1] = (main.prefix_sums[l3][idx] + (A[idx] if lengths[idx] == l3 else 0)) % MOD
            # Now, for the current i, the sum of A_j with length l2 and j > i is
            # prefix_sums[l2][N] - prefix_sums[l2][i+1]
            sum_j = (main.prefix_sums[l2][N] - main.prefix_sums[l2][i+1]) % MOD
            # The contribution is a * 10^l2 * sum_j
            contribution = (a * power[l2] % MOD) * sum_j % MOD
            total = (total + contribution) % MOD
    
    print(total)

if __name__ == "__main__":
    main()