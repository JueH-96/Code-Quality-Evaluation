MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:2+N]))
    
    # Count the number of -1s
    q = B.count(-1)
    
    # Precompute the number of possible sequences
    total_sequences = pow(M, q, MOD)
    
    # The number of connected components is always N - (number of edges)
    # But in this problem, the number of connected components is the number of distinct connected components in the graph
    # For the given problem, the number of connected components is the number of distinct connected components in the graph
    # which is the number of distinct connected components in the graph formed by the edges (i,j) where A_i <= A_j
    
    # To compute the sum of f(B') over all possible B', we need to find the expected value of f(B') and multiply by the total number of sequences
    
    # The expected value of f(B') is the expected number of connected components in the graph
    
    # The expected number of connected components is N minus the expected number of edges
    
    # The expected number of edges is the sum over all pairs (i,j) of the probability that A_i <= A_j
    
    # So, the expected number of connected components is N - sum_{i<j} P(A_i <= A_j)
    
    # Now, we need to compute the sum of P(A_i <= A_j) over all pairs (i,j)
    
    # For each pair (i,j), if both B_i and B_j are fixed, then P(A_i <= A_j) is 1 if B_i <= B_j, else 0
    
    # If one of them is -1, then we need to compute the probability that A_i <= A_j
    
    # If both are -1, then the probability is (M+1)/(2M)
    
    # So, we need to handle these cases
    
    # First, precompute the positions of -1s
    minus_one_indices = [i for i, x in enumerate(B) if x == -1]
    
    # Precompute the positions of fixed elements
    fixed_indices = [i for i, x in enumerate(B) if x != -1]
    
    # Initialize the sum of P(A_i <= A_j)
    sum_p = 0
    
    # Handle pairs where both are fixed
    for i in fixed_indices:
        for j in fixed_indices:
            if i < j:
                if B[i] <= B[j]:
                    sum_p += 1
    
    # Handle pairs where one is fixed and one is -1
    for i in fixed_indices:
        for j in minus_one_indices:
            if i < j:
                # P(A_i <= A_j) = (M - B[i] + 1) / M if B[i] <= M
                if B[i] <= M:
                    sum_p += (M - B[i] + 1) / M
                else:
                    sum_p += 0
            elif j < i:
                # P(A_j <= A_i) = B[i] / M
                sum_p += B[i] / M
    
    # Handle pairs where both are -1
    for i in minus_one_indices:
        for j in minus_one_indices:
            if i < j:
                # P(A_i <= A_j) = (M + 1) / (2 * M)
                sum_p += (M + 1) / (2 * M)
    
    # The expected number of connected components is N - sum_p
    expected_f = N - sum_p
    
    # The total sum is expected_f * total_sequences
    total_sum = int(expected_f * total_sequences) % MOD
    
    print(total_sum)

if __name__ == "__main__":
    main()