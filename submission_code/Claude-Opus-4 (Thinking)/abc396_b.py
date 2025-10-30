# YOUR CODE HERE
# Read the number of queries
Q = int(input())

# Initialize the stack with 100 cards labeled 0
# The top of the stack is at the end of the list
stack = [0] * 100

# Process queries
for _ in range(Q):
    query = input().split()
    query_type = int(query[0])
    
    if query_type == 1:
        # Type 1: Place a card labeled x on top
        x = int(query[1])
        stack.append(x)
    else:
        # Type 2: Remove the top card and output its value
        top_card = stack.pop()
        print(top_card)