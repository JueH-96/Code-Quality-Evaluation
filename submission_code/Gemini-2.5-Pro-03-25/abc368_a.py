import sys

# Read N and K from the first line of stdin
line1 = sys.stdin.readline().split()
n = int(line1[0])
k = int(line1[1])

# Read the list of integers A from the second line of stdin
a = list(map(int, sys.stdin.readline().split()))

# Check if K is 0. Although the constraint says 1 <= K,
# if K were 0, no operation would happen, and the list remains the same.
# However, given the constraint 1 <= K < N, this case is not possible.

# The operation involves taking the last K elements and moving them to the front.
# We can split the list A into two parts:
# 1. The first N-K elements (the ones that remain at the bottom after the move)
# 2. The last K elements (the ones that are moved to the top)

# Extract the last K elements (bottom K cards)
# In Python slicing, a[start:] takes elements from index 'start' to the end.
# The indices of the last K elements start from N-K.
# Example: N=5, K=3, indices are 0, 1, 2, 3, 4. Last 3 elements are at indices 2, 3, 4.
# N-K = 5-3 = 2. So a[2:] gives elements at index 2, 3, 4.
bottom_k_cards = a[n-k:]

# Extract the first N-K elements (top N-K cards initially, become bottom N-K cards)
# In Python slicing, a[:end] takes elements from the beginning up to (but not including) index 'end'.
# We need the first N-K elements, which are at indices 0 to N-K-1.
# a[:n-k] gives elements at indices 0, 1, ..., n-k-1.
remaining_cards = a[:n-k]

# Construct the new stack by placing the bottom_k_cards on top of the remaining_cards.
# The '+' operator concatenates lists in Python.
new_stack = bottom_k_cards + remaining_cards

# Print the elements of the new stack, separated by spaces.
# The `*` operator unpacks the list elements as arguments to print.
# `print` by default separates arguments with spaces and adds a newline at the end.
print(*new_stack)