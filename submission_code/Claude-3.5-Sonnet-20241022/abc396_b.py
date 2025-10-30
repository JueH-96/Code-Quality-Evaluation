# Initialize stack with 100 zeros
stack = [0] * 100

# Read number of queries
Q = int(input())

# Process each query
for _ in range(Q):
    query = input().split()
    query_type = int(query[0])
    
    if query_type == 1:
        # Add card with value x on top
        x = int(query[1])
        stack.append(x)
    else:  # query_type == 2
        # Remove and print top card
        print(stack.pop())