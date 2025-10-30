# YOUR CODE HERE
Q = int(input())
A = []
for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        x = int(query[1])
        A.append(x)
    elif query[0] == '2':
        k = int(query[1])
        print(A[-k])