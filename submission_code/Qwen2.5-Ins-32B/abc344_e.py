# YOUR CODE HERE
import sys

input = sys.stdin.read
data = input().split()

idx = 0
N = int(data[idx])
idx += 1
A = list(map(int, data[idx:idx+N]))
idx += N
Q = int(data[idx])
idx += 1

# Create a dictionary to store the index of each element in A
index_map = {val: i for i, val in enumerate(A)}

# Convert A to a list of tuples (value, index)
A = [(val, i) for i, val in enumerate(A)]

for _ in range(Q):
    query_type = int(data[idx])
    idx += 1
    if query_type == 1:
        x = int(data[idx])
        idx += 1
        y = int(data[idx])
        idx += 1
        # Find the index of x in A
        x_index = index_map[x]
        # Insert y immediately after x
        new_index = x_index + 1
        A.insert(new_index, (y, new_index))
        # Update the index_map for all elements after x
        for i in range(new_index + 1, len(A)):
            index_map[A[i][0]] += 1
        index_map[y] = new_index
    else:
        x = int(data[idx])
        idx += 1
        # Find the index of x in A
        x_index = index_map[x]
        # Remove x from A
        A.pop(x_index)
        # Update the index_map for all elements after x
        for i in range(x_index, len(A)):
            index_map[A[i][0]] -= 1
        del index_map[x]

# Extract the values from A
result = [val for val, _ in A]
print(' '.join(map(str, result)))