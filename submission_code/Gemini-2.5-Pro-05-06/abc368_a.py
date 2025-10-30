# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# The K cards from the bottom of the stack.
# In a 0-indexed list A of length N, these are elements from index N-K to N-1.
# Slicing A[N-K:] captures these elements.
cards_from_bottom = A[N-K:]

# The remaining N-K cards that were originally at the top of the stack.
# These are elements from index 0 to N-K-1.
# Slicing A[:N-K] captures these elements.
cards_at_top_initially = A[:N-K]

# The new stack is formed by placing `cards_from_bottom` on top of `cards_at_top_initially`.
# Both segments maintain their internal order.
new_stack = cards_from_bottom + cards_at_top_initially

# Print the integers in the new_stack, separated by spaces.
# The * operator unpacks the list `new_stack` into individual arguments for print,
# which then prints them separated by spaces by default.
print(*(new_stack))