from collections import deque

# Read the input
N = int(input())
A = list(map(int, input().split()))
Q = int(input())

# Create a deque to store the sequence
seq = deque(A)

# Process the queries
for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        x, y = map(int, query[1:])
        idx = seq.index(x)
        seq.insert(idx+1, y)
    else:
        x = int(query[1])
        seq.remove(x)

# Print the final sequence
print(*seq)