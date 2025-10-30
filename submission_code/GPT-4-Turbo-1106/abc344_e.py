# Read the initial sequence length and the sequence itself
N = int(input())
A = list(map(int, input().split()))

# Read the number of queries
Q = int(input())

# Process each query
for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        # Insert y immediately after the element x
        x, y = query[1], query[2]
        idx = A.index(x)
        A.insert(idx + 1, y)
    elif query[0] == 2:
        # Remove the element x
        x = query[1]
        A.remove(x)

# Print the final sequence
print(' '.join(map(str, A)))