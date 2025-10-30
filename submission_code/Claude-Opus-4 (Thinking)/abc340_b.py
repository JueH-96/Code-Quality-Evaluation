# YOUR CODE HERE
Q = int(input())
A = []

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Append x to the end of A
        x = query[1]
        A.append(x)
    else:  # query[0] == 2
        # Find the k-th value from the end
        k = query[1]
        print(A[-k])