import sys

# Read the number of elements N
N = int(sys.stdin.readline())

# Read the N integers into a list
S = list(map(int, sys.stdin.readline().split()))

# Sort the list. This is crucial for the two-pointer approach.
# Sorting takes O(N log N) time.
s_list = sorted(S)

# Initialize the count of fine triplets
count = 0

# A fine triplet (A, B, C) from S satisfies A < B < C and B - A = C - B.
# This is equivalent to A + C = 2 * B.
# We can iterate through each element s_list[j] as the potential middle element B.
# For a fixed s_list[j], we need to find pairs (s_list[i], s_list[k]) from the list
# such that i < j < k and s_list[i] + s_list[k] = 2 * s_list[j].
# Since the list is sorted, i < j automatically implies s_list[i] < s_list[j],
# and k > j automatically implies s_list[k] > s_list[j].
# The condition A < B < C is satisfied if i < j < k.
# The middle element s_list[j] cannot be the smallest or the largest in a triplet
# from the set S (since A, B, C are distinct), so j can range from 1 to N-2.
for j in range(1, N - 1):
    # The target sum for the pair (A, C) is 2 * s_list[j]
    target_sum = 2 * s_list[j]
    
    # Initialize two pointers for the elements A and C:
    # The left pointer starts at the beginning of the list (index 0)
    left = 0
    # The right pointer starts at the end of the list (index N-1)
    right = N - 1

    # Use two pointers to find pairs (s_list[left], s_list[right])
    # that sum up to target_sum.
    # The constraints left < j and right > j ensure that s_list[left]
    # is considered from the part of the list before s_list[j]
    # and s_list[right] is considered from the part of the list after s_list[j].
    while left < j and right > j:
        current_sum = s_list[left] + s_list[right]

        if current_sum == target_sum:
            # Found a triplet (s_list[left], s_list[j], s_list[right]).
            # A = s_list[left], B = s_list[j], C = s_list[right].
            # Since s_list is sorted and left < j < right, we have s_list[left] < s_list[j] < s_list[right].
            # The condition A < B < C is satisfied.
            # The condition A + C = 2B is satisfied by current_sum == target_sum.
            count += 1
            
            # Since all elements in S are distinct, the current s_list[left]
            # cannot form another fine triplet with s_list[j] by changing right,
            # and similarly for s_list[right] by changing left.
            # We need to move both pointers inwards to look for other potential pairs
            # (s_list[left'], s_list[right']) that sum to target_sum.
            left += 1
            right -= 1
            
        elif current_sum < target_sum:
            # The current sum s_list[left] + s_list[right] is too small.
            # To increase the sum, we need a larger value from the left side.
            # Move the left pointer to the right to consider a larger A.
            left += 1
        else: # current_sum > target_sum
            # The current sum s_list[left] + s_list[right] is too large.
            # To decrease the sum, we need a smaller value from the right side.
            # Move the right pointer to the left to consider a smaller C.
            right -= 1

# Print the total number of fine triplets found
print(count)