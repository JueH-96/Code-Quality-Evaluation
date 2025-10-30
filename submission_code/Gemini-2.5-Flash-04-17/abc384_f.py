import sys

# Helper function to count trailing zeros (nu2)
# Constraints A_i >= 1, so x = A_i or x = A_i+A_j >= 2. x is always positive.
def nu2(x):
    # Use bitwise operations for efficiency
    # x & -x isolates the rightmost set bit (which is 2^k where k is nu2(x))
    # bit_length() gives the position of the most significant bit (1-indexed)
    # For y = 2^k, bit_length() = k+1. So nu2(x) = log2(x & -x).bit_length() - 1.
    return (x & -x).bit_length() - 1

# Helper function to calculate f(x)
# f(x) is x divided by the largest power of 2 that divides x
def f(x):
    # Constraints A_i >= 1, so x = A_i or x = A_i+A_j >= 2. x is always positive.
    # Equivalent to x // (1 << nu2(x))
    shift = nu2(x)
    return x >> shift

def solve():
    # Read input
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the first part of the sum: sum_{i=1}^N f(A_i+A_i) = sum_{i=1}^N f(2A_i) = sum_{i=1}^N f(A_i)
    sum_f_A = 0
    for x in A:
        sum_f_A += f(x)

    # Calculate the second part of the sum related to sum_{i=1}^N sum_{j=1}^N f(A_i+A_j)
    # Let T = sum_{i=1}^N sum_{j=1}^N f(A_i+A_j).
    # The target sum S = sum_{i<=j} f(A_i+A_j).
    # We know T = sum_{i} f(A_i) + 2 * sum_{i<j} f(A_i+A_j).
    # And S = sum_{i} f(A_i) + sum_{i<j} f(A_i+A_j).
    # Thus, T = sum_{i} f(A_i) + 2 * (S - sum_{i} f(A_i)) = 2S - sum_{i} f(A_i).
    # So, 2S = T + sum_{i} f(A_i).
    # We will compute T and sum_f_A, then calculate S.

    # T = sum_{i,j} f(A_i+A_j) = sum_{k=0...max_k} sum_{i,j} I(nu2(A_i+A_j)=k) * (A_i+A_j)/2^k
    # The condition nu2(X)=k is equivalent to X % 2^(k+1) == 2^k for X > 0.
    
    T = 0
    # Maximum possible value of A_i + A_j is 2 * 10^7.
    # The maximum power of 2 that can divide a number <= 2*10^7 is 2^24 = 16777216.
    # So nu2(A_i+A_j) can be at most 24. We loop k from 0 to 24.
    MAX_K = 24 

    for k in range(MAX_K + 1):
        M = 1 << (k + 1) # Modulo 2^(k+1)
        
        # Compute counts and sums of A_i modulo M using dictionaries.
        # Dictionary keys are remainders in [0, M-1], values are counts/sums.
        # The number of distinct remainders is at most N.
        cnt = {} # remainder -> count
        sum_orig = {} # remainder -> sum of original values
        
        for x in A:
            r = x % M
            cnt[r] = cnt.get(r, 0) + 1
            sum_orig[r] = sum_orig.get(r, 0) + x

        # We need pairs (v1, v2) such that v1 + v2 = 2^k or v1 + v2 = 2^k + M.
        # Iterate over distinct remainders v1 that appear in A mod M (keys in cnt).
        
        # Target sum 1: 2^k
        target_sum1 = 1 << k 
        for v1 in cnt:
            v2 = target_sum1 - v1
            # Check if v2 is a remainder present in A mod M
            if v2 in cnt:
                # Add sum(A_i+A_j) / 2^k for pairs (i,j) with A_i%M=v1, A_j%M=v2.
                # There are cnt[v1] * cnt[v2] such pairs of indices.
                # The sum of A_i+A_j over these pairs is sum(A_i where A_i%M=v1) * count(A_j where A_j%M=v2) + sum(A_j where A_j%M=v2) * count(A_i where A_i%M=v1)
                # = sum_orig[v1] * cnt[v2] + sum_orig[v2] * cnt[v1].
                T += (sum_orig[v1] * cnt[v2] + sum_orig[v2] * cnt[v1]) // (1 << k)
        
        # Target sum 2: 2^k + M
        # This sum is only possible if 2^k + M < 2 * M (max sum of remainders is 2*(M-1)), which is true for k >= 1.
        # If k=0, M=2. target_sum2 = 1+2=3. Max v1+v2 = 1+1=2. Impossible.
        if k >= 1:
            target_sum2 = (1 << k) + M 
            for v1 in cnt:
                 v2 = target_sum2 - v1
                 # Check if v2 is a remainder present in A mod M
                 if v2 in cnt:
                      T += (sum_orig[v1] * cnt[v2] + sum_orig[v2] * cnt[v1]) // (1 << k)

    # The sum T calculated is sum_{i=1}^N sum_{j=1}^N f(A_i+A_j).
    # The required answer is S = sum_{i=1}^N sum_{j=i}^N f(A_i+A_j).
    # We use the relationship 2S = T + sum_{i=1}^N f(A_i).
    # The result must be an integer, so integer division is appropriate.
    print((T + sum_f_A) // 2)

solve()