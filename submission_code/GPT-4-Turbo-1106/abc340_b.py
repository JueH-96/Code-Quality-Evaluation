# Read the number of queries
Q = int(input())

# Initialize the sequence A
A = []

# Process each query
for _ in range(Q):
    query = input().split()
    type_of_query = int(query[0])
    
    if type_of_query == 1:
        # Append x to the end of A
        x = int(query[1])
        A.append(x)
    elif type_of_query == 2:
        # Find the k-th value from the end of A
        k = int(query[1])
        print(A[-k])