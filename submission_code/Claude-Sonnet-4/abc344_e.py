# Read initial sequence
N = int(input())
A = list(map(int, input().split()))

# Read number of queries
Q = int(input())

# Process each query
for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:  # Insert y after x
        x, y = query[1], query[2]
        # Find index of x
        idx = A.index(x)
        # Insert y immediately after x (at position idx + 1)
        A.insert(idx + 1, y)
    
    elif query[0] == 2:  # Remove x
        x = query[1]
        # Remove x from the list
        A.remove(x)

# Print the final sequence
print(*A)