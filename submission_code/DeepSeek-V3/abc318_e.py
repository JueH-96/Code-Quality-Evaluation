# YOUR CODE HERE
from collections import defaultdict

def count_valid_triples(N, A):
    # Precompute the total number of elements equal to each value
    total_counts = defaultdict(int)
    for num in A:
        total_counts[num] += 1
    
    # Precompute the prefix sums for each value
    prefix = defaultdict(list)
    for idx, num in enumerate(A):
        prefix[num].append(idx + 1)  # +1 to convert to 1-based index
    
    # Initialize the result
    result = 0
    
    # Iterate through each unique value in A
    for num in prefix:
        if total_counts[num] < 2:
            continue  # Need at least two occurrences to form a triple
        indices = prefix[num]
        # Precompute the prefix sum of the counts of elements not equal to num
        # We need to find the number of elements not equal to num between each pair of indices
        # To do this, we can precompute the prefix sum of the counts of elements not equal to num
        # and then use it to calculate the number of such elements between any two indices
        # First, create a list that marks the positions of elements not equal to num
        # We can create a list where for each index, we mark if the element is not equal to num
        # Then, we can compute the prefix sum of this list
        # Initialize a list to mark positions where A[i] != num
        not_num = [0] * (N + 2)
        for i in range(1, N+1):
            if A[i-1] != num:
                not_num[i] = 1
        # Compute the prefix sum of not_num
        prefix_not_num = [0] * (N + 2)
        for i in range(1, N+1):
            prefix_not_num[i] = prefix_not_num[i-1] + not_num[i]
        # Now, for each pair of indices (i, k) where A[i] == A[k] == num and i < k
        # We need to find the number of j where i < j < k and A[j] != num
        # This is equal to prefix_not_num[k-1] - prefix_not_num[i]
        # We can iterate through all pairs of indices in the prefix[num] list
        # and sum the above value for each pair
        # Since the indices are sorted, we can use two pointers or a nested loop
        # However, with N up to 3e5, a nested loop would be too slow
        # Instead, we can precompute the prefix sum of the counts of elements not equal to num
        # and then for each pair (i, k), compute the number of j in between
        # To optimize, we can precompute the prefix sum of the counts of elements not equal to num
        # and then for each pair (i, k), the number of j is prefix_not_num[k-1] - prefix_not_num[i]
        # Since the indices are sorted, we can iterate through the list and for each i, iterate through all k > i
        # But this is still O(n^2) in the worst case
        # To further optimize, we can note that for each i, the number of k is the number of elements in the list after i
        # and for each k, the number of j is prefix_not_num[k-1] - prefix_not_num[i]
        # So, for each i, we can compute the sum of (prefix_not_num[k-1] - prefix_not_num[i]) for all k > i
        # This can be rewritten as sum_{k > i} prefix_not_num[k-1] - prefix_not_num[i] * (number of k > i)
        # So, we can precompute the sum of prefix_not_num[k-1] for all k > i
        # To do this, we can precompute the suffix sum of prefix_not_num
        # Let's compute the suffix sum of prefix_not_num
        suffix_sum = [0] * (N + 2)
        for i in range(N, 0, -1):
            suffix_sum[i] = suffix_sum[i+1] + prefix_not_num[i]
        # Now, for each i in the indices list, we can compute the sum of prefix_not_num[k-1] for all k > i
        # as suffix_sum[i+1] - suffix_sum[k_max + 1], where k_max is the last index in the list
        # Wait, no. The suffix_sum[i] is the sum of prefix_not_num[i..N]
        # So, for each i, the sum of prefix_not_num[k-1] for k > i is suffix_sum[i+1] - suffix_sum[k_max + 1]
        # But since k_max is the last index, suffix_sum[k_max + 1] is 0
        # So, the sum is suffix_sum[i+1]
        # Therefore, for each i, the sum of prefix_not_num[k-1] for k > i is suffix_sum[i+1]
        # So, the total for each i is suffix_sum[i+1] - prefix_not_num[i] * (number of k > i)
        # The number of k > i is len(indices) - (i's position + 1)
        # Wait, no. The indices list is sorted, so for each i in the list, the number of k > i is len(indices) - (position of i + 1)
        # So, for each i in the list, the number of k > i is len(indices) - (current index + 1)
        # So, we can iterate through the indices list and for each i, compute the number of k > i
        # and then compute the sum as suffix_sum[i+1] - prefix_not_num[i] * (number of k > i)
        # So, let's implement this
        for idx, i in enumerate(indices):
            # Number of k > i
            num_k = len(indices) - (idx + 1)
            if num_k == 0:
                continue
            # Sum of prefix_not_num[k-1] for k > i
            sum_prefix = suffix_sum[i+1]
            # Total for this i is sum_prefix - prefix_not_num[i] * num_k
            result += sum_prefix - prefix_not_num[i] * num_k
    return result

# Read input
N = int(input())
A = list(map(int, input().split()))
# Compute and print the result
print(count_valid_triples(N, A))