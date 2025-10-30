import sys
input = sys.stdin.read

def find_triplet():
    data = input().split()
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))
    
    # Dictionary to store pairs and their sums
    pair_sums = {}
    
    # We need to find i, j, k such that 1 <= i < j < k <= N and A[i] + A[j] + A[k] = X
    # We can iterate over k and check for each k if there exists i, j such that A[i] + A[j] = X - A[k]
    
    # We will iterate over k from 3 to N (1-based index)
    for k in range(2, N):
        # Before checking for k, we can add all pairs (i, j) for j = k to the dictionary
        # We do this because we need pairs (i, j) where j < k
        for i in range(k):
            sum_ij = A[i] + A[k-1]
            if sum_ij not in pair_sums:
                pair_sums[sum_ij] = []
            pair_sums[sum_ij].append((i+1, k))  # Store 1-based indices
        
        # Now check if there exists any pair (i, j) such that A[i] + A[j] = X - A[k+1]
        required_sum = X - A[k]
        if required_sum in pair_sums:
            # We have one or more pairs (i, j) such that i < j < k+1 (1-based index)
            for (i, j) in pair_sums[required_sum]:
                if j < k+1:  # Ensure j < k (1-based index)
                    print(i, j, k+1)
                    return
    
    # If no such triplet is found
    print(-1)

find_triplet()