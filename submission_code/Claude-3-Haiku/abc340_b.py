# Read the number of queries
Q = int(input())

# Initialize the sequence A
A = []

# Process the queries
for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        # Append the value to the end of A
        A.append(int(query[1]))
    else:
        # Find the k-th value from the end of A
        k = int(query[1])
        print(A[-k])