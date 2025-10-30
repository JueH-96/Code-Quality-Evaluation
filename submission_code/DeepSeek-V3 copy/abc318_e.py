from collections import defaultdict

def count_valid_triples(N, A):
    # Create a dictionary to store the indices of each element
    element_indices = defaultdict(list)
    for idx, val in enumerate(A):
        element_indices[val].append(idx)
    
    total = 0
    for val in element_indices:
        indices = element_indices[val]
        if len(indices) < 2:
            continue
        # Precompute the prefix sum of the counts of elements not equal to val
        # We need to find the number of elements not equal to val between each pair of indices
        # To do this, we can precompute the prefix sum of the counts of elements not equal to val
        # Then, for each pair (i, k), the number of j is the number of elements not equal to val between i and k
        # Which is prefix[k] - prefix[i] - (number of elements equal to val between i and k)
        # But since we are only considering pairs where A_i = A_k, the number of elements equal to val between i and k is the number of indices of val between i and k
        # So, the number of j is prefix[k] - prefix[i] - (number of indices of val between i and k)
        # To compute this efficiently, we need to precompute the prefix sum of the counts of elements not equal to val
        # And also precompute the prefix sum of the counts of elements equal to val
        # So, we create two prefix sums:
        # prefix_not_val: prefix sum of counts of elements not equal to val
        # prefix_val: prefix sum of counts of elements equal to val
        prefix_not_val = [0] * (N + 1)
        prefix_val = [0] * (N + 1)
        for i in range(N):
            prefix_not_val[i+1] = prefix_not_val[i] + (1 if A[i] != val else 0)
            prefix_val[i+1] = prefix_val[i] + (1 if A[i] == val else 0)
        # Now, for each pair of indices (i, k) where i < k and A[i] == A[k] == val
        # The number of j is prefix_not_val[k] - prefix_not_val[i+1]
        # Because j must be between i and k, and A[j] != val
        # So, for each pair (i, k), the number of j is prefix_not_val[k] - prefix_not_val[i+1]
        # We need to sum this over all pairs (i, k)
        # To do this efficiently, we can precompute the sum of prefix_not_val[k] for all k in indices
        # And the sum of prefix_not_val[i+1] for all i in indices
        # Then, the total is sum_{i < k} (prefix_not_val[k] - prefix_not_val[i+1]) = sum_{k} prefix_not_val[k] * (number of i < k) - sum_{i} prefix_not_val[i+1] * (number of k > i)
        # Since the number of i < k is the index of k in the indices list
        # And the number of k > i is the length of indices minus the index of i minus 1
        # So, we can compute the total as follows:
        sum_prefix_k = 0
        sum_prefix_i_plus_1 = 0
        for idx, k in enumerate(indices):
            sum_prefix_k += prefix_not_val[k] * idx
        for idx, i in enumerate(indices):
            sum_prefix_i_plus_1 += prefix_not_val[i+1] * (len(indices) - idx - 1)
        total += sum_prefix_k - sum_prefix_i_plus_1
    return total

# Read input
N = int(input())
A = list(map(int, input().split()))
# Compute and print the result
print(count_valid_triples(N, A))