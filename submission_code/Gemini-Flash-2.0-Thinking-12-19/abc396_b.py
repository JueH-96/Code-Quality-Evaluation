# YOUR CODE HERE
import sys

# Read Q, the number of queries
Q = int(sys.stdin.readline())

# Use a Python list to simulate the stack.
# The list `pushed_cards` stores the cards that have been explicitly placed
# on top of the initial stack using Type 1 queries. The top of the stack
# corresponds to the last element of this list.
# The initial 100 cards labeled 0 are conceptually located below the cards
# in `pushed_cards`. They are accessed only when `pushed_cards` is empty.
pushed_cards = []

# Process each query
for _ in range(Q):
    # Read the query line and split it into parts
    line = sys.stdin.readline().split()
    query_type = int(line[0])

    if query_type == 1:
        # Type 1 query: Place a card labeled x on top of the stack.
        # The value x is the second part of the query line.
        x = int(line[1])
        # Add the card x to the top of the conceptual stack, which is the end of our list.
        pushed_cards.append(x)
    elif query_type == 2:
        # Type 2 query: Remove the top card and output its integer label.
        # The problem guarantees that the stack always has at least one card
        # when a type 2 query occurs.
        if pushed_cards:
            # If the list of explicitly pushed cards is not empty, the top card
            # is one that was added by a Type 1 query. Remove and print it.
            # list.pop() removes and returns the last element (the top).
            print(pushed_cards.pop())
        else:
            # If the list of explicitly pushed cards is empty, it means all cards
            # added by Type 1 queries have been removed. The top card must now be
            # one of the initial 100 cards, all labeled with 0.
            # Since the problem guarantees the stack is not empty for Type 2 queries,
            # and we started with 100 cards, if pushed_cards is empty, the top must be a 0.
            # We don't need to simulate the initial 100 cards explicitly, just output 0.
            print(0)