import sys

# Read the number of queries
Q = int(sys.stdin.readline())

# Initialize the stack. The problem states there are 100 cards
# labeled with 0 initially. We represent the stack as a list
# where the end of the list is the top of the stack.
stack = [0] * 100

# Process each query
for _ in range(Q):
    # Read the query line and split it into parts
    query = sys.stdin.readline().split()
    query_type = int(query[0])

    if query_type == 1:
        # Type 1: Place a card with value x on top of the stack.
        # x is the second element in the query list (as string), convert to int.
        x = int(query[1])
        # Add the new card to the end of the list (top of the stack).
        stack.append(x)
    elif query_type == 2:
        # Type 2: Remove the top card and output its value.
        # The problem guarantees the stack always has at least one card
        # when a type 2 query occurs, so pop is safe.
        # Remove and get the last element from the list (top of the stack).
        removed_card = stack.pop()
        # Output the value of the removed card.
        print(removed_card)