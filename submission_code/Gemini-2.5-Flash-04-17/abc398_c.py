import sys
from collections import Counter

# Read N
N = int(sys.stdin.readline())

# Read the list of integers A
# Constraints guarantee N >= 1 and the line will contain N integers.
# rstrip() is good practice to remove potential trailing newline.
line = sys.stdin.readline().rstrip()
A = list(map(int, line.split()))

# Compute frequency of each integer
# This is efficient O(N) on average using hashing
counts = Counter(A)

# Find the maximum unique value and corresponding person's label
# Initialize max_unique_value to a value smaller than any possible A_i (which are >= 1)
# Using -1 works because any valid A_i (>= 1) will be greater than -1.
max_unique_value = -1
# Initialize result_label to -1, which is the required output if no person satisfies the condition.
result_label = -1

# Iterate through the list A with original indices (labels are i + 1)
# This pass identifies persons whose value is unique and finds the one
# with the maximum value among them.
for i in range(N):
    current_value = A[i]
    
    # Check if the current integer appears exactly once in the entire list
    # We can safely access counts[current_value] because current_value comes from A,
    # which was used to build the counter. Counter returns 0 for unseen elements,
    # but all elements from A were used to build the counter.
    if counts[current_value] == 1:
        # This person (with label i+1) has a unique integer.
        # Check if this unique integer is strictly greater than the maximum unique integer found so far.
        if current_value > max_unique_value:
            max_unique_value = current_value # Update the maximum unique value found
            result_label = i + 1 # Store the 1-based label of this person

# After iterating through all people, result_label holds:
# - The label of the person with the largest unique integer, if at least one such person exists.
# - -1, if no person had a unique integer (because max_unique_value would remain -1).
print(result_label)