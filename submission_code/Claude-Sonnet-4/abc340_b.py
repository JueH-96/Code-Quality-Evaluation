# YOUR CODE HERE
Q = int(input())
A = []

for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        # Type 1: append x to end of A
        x = int(query[1])
        A.append(x)
    else:
        # Type 2: find k-th value from end of A
        k = int(query[1])
        # k-th from end means A[-k]
        print(A[-k])