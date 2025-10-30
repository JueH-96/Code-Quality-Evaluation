import sys
import itertools

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    if K == 1:
        print(max(A))
        return
    
    # Since binomial coefficient is at most 10^6, we can afford to generate combinations
    # when K is small enough compared to N, but not for large K.
    # We need to find a way to compute the maximum XOR without generating all combinations.
    
    # We will use a greedy approach with basis reduction in a vector space over GF(2)
    # This is similar to finding the maximum XOR subset using Gaussian elimination in GF(2)
    
    # Basis for the maximum xor we can achieve
    basis = [0] * 60  # since 0 <= A_i < 2^60
    
    # Function to insert a number into the basis
    def insert_number(x):
        for i in range(59, -1, -1):
            if (x >> i) & 1:
                if not basis[i]:
                    basis[i] = x
                    return
                x ^= basis[i]
    
    # Build the basis
    for num in A:
        insert_number(num)
    
    # Now, use the basis to construct the maximum XOR of exactly K elements
    # Start from the highest basis element and try to include exactly K elements
    # We need to use a dynamic programming approach to check if we can form a subset of exactly K elements
    # dp[j] will be True if we can form a subset with exactly j elements
    dp = [False] * (K + 1)
    dp[0] = True  # We can always form an empty subset
    
    for b in basis:
        if b == 0:
            continue
        # Update dp in reverse to avoid overwriting issues
        for j in range(K, 0, -1):
            if dp[j - 1]:
                dp[j] = True
    
    # If we cannot form a subset of exactly K elements, output is not possible
    if not dp[K]:
        print(0)
        return
    
    # If possible, construct the maximum XOR of exactly K elements
    result = 0
    count = 0
    for i in range(59, -1, -1):
        if basis[i] and (result ^ basis[i]) > result and count < K:
            result ^= basis[i]
            count += 1
    
    print(result)

if __name__ == "__main__":
    main()