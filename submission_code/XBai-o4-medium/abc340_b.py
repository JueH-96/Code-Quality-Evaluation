# Read the number of queries
Q = int(input())
# Initialize the sequence as an empty list
A = []
# Process each query
for _ in range(Q):
    parts = input().split()
    if parts[0] == '1':
        # Append x to the end of A
        x = int(parts[1])
        A.append(x)
    else:
        # Find the k-th value from the end
        k = int(parts[1])
        print(A[-k])