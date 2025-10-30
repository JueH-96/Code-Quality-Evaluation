import sys

def solve():
    """
    Solves the problem by calculating the sum in a single pass.
    """
    MOD = 998244353
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    total_sum = 0
    prefix_sum = 0
    
    # We compute the total sum S = sum_{i<j} f(A_i, A_j)
    # This can be rewritten by changing the order of summation:
    # S = sum_{j=1..N-1} sum_{i=0..j-1} f(A_i, A_j)
    # The inner sum is the contribution of all pairs ending with A_j.
    # sum_{i<j} f(A_i, A_j) = sum_{i<j} (A_i * 10^d_j + A_j)
    #                        = (sum_{i<j} A_i) * 10^d_j + j * A_j
    #                        = prefix_sum_at_{j-1} * 10^d_j + j * A_j
    
    # We iterate i from 0 to N-1, representing the second element of the pair.
    for i in range(N):
        # When i > 0, we calculate the contribution of pairs (k, i) where k < i.
        if i > 0:
            # Number of digits in the current element A[i]
            num_digits_i = len(str(A[i]))
            
            # Contribution from the "left part" of the concatenation.
            # This corresponds to (sum_{k<i} A_k) * 10^{d_i}
            term1 = (prefix_sum * pow(10, num_digits_i, MOD)) % MOD
            
            # Contribution from the "right part" of the concatenation.
            # This corresponds to sum_{k<i} A_i, which is A_i added i times.
            term2 = (i * A[i]) % MOD
            
            total_sum = (total_sum + term1 + term2) % MOD
            
        # Update the prefix sum for the next iteration.
        prefix_sum = (prefix_sum + A[i]) % MOD

    print(total_sum)

solve()