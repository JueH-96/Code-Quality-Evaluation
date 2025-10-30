import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0

# Read the number of queries Q
Q = int(data[index])
index += 1

# Initialize an empty list A
A = []

# Process each query
for _ in range(Q):
    query_type = int(data[index])
    index += 1
    if query_type == 1:
        # Append x to the end of A
        x = int(data[index])
        index += 1
        A.append(x)
    elif query_type == 2:
        # Find the k-th value from the end and print it
        k = int(data[index])
        index += 1
        print(A[-k])