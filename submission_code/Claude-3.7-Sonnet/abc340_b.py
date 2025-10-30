# Read the number of queries
Q = int(input())

# Initialize an empty sequence
A = []

# Process each query
for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        x = int(query[1])
        A.append(x)
    else:  # query[0] == '2'
        k = int(query[1])
        print(A[-k])