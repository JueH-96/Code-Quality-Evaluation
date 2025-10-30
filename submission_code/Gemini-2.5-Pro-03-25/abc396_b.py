# YOUR CODE HERE
import sys

# Initialize the stack with 100 cards labeled 0
# We use a Python list to represent the stack.
# The end of the list represents the top of the stack.
stack = [0] * 100

# Read the number of queries
q = int(sys.stdin.readline())

# Process each query
for _ in range(q):
    # Read the query line and split it into parts
    line = sys.stdin.readline().split()
    # The first part is the query type
    query_type = int(line[0])

    if query_type == 1:
        # Type 1 query: Place a card with label x on top
        # The value x is the second part of the query line
        x = int(line[1])
        # Append adds the element to the end (top) of the list (stack)
        stack.append(x)
    elif query_type == 2:
        # Type 2 query: Remove the top card and output its label
        # The problem guarantees the stack will have at least one card,
        # so pop() will not raise an IndexError.
        # pop() removes and returns the last element (top) of the list (stack)
        removed_card = stack.pop()
        # Print the label of the removed card
        print(removed_card)

# END OF YOUR CODE HERE