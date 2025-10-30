# Read the number of queries
Q = int(input())

# Initialize an empty list to store the sequence
A = []

# Process each query
for _ in range(Q):
    # Read the query
    query = input().split()

    # If the query is of type 1, append the value to the end of A
    if query[0] == '1':
        A.append(int(query[1]))
    # If the query is of type 2, find the k-th value from the end of A
    elif query[0] == '2':
        k = int(query[1])
        # Print the k-th value from the end of A
        print(A[-k])