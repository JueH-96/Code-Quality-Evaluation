# YOUR CODE HERE
def find_triplet(N, X, A):
    # We will use a dictionary to store pairs of numbers and their indices
    # This will help us to find a triplet in O(N^2) time complexity
    pair_sums = {}

    # Iterate over each pair (i, j) with i < j
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate the sum of the current pair
            pair_sum = A[i] + A[j]
            # Calculate the required third number to reach X
            required = X - pair_sum

            # Check if the required number is already in the dictionary
            if required in pair_sums:
                # Check if the indices are valid (i < j < k)
                for k in pair_sums[required]:
                    if k > j:
                        # Return the 1-based indices
                        return i + 1, j + 1, k + 1

            # Add the current pair (i, j) to the dictionary
            if pair_sum not in pair_sums:
                pair_sums[pair_sum] = []
            pair_sums[pair_sum].append(j)

    # If no valid triplet is found, return -1
    return -1

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
A = list(map(int, data[2:]))

result = find_triplet(N, X, A)
if result == -1:
    print(-1)
else:
    print(result[0], result[1], result[2])