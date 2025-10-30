# Read the number of queries
Q = int(input())

# Initialize the stack with 100 cards labeled 0
stack = [0] * 100

# Initialize the output list
output = []

# Process the queries
for _ in range(Q):
    query = input().split()
    query_type = int(query[0])
    
    if query_type == 1:
        # Add a card labeled with the given integer on top of the stack
        stack.append(int(query[1]))
    else:
        # Remove the top card and output the integer written on it
        output.append(stack.pop())

# Print the output
for num in output:
    print(num)