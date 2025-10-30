# YOUR CODE HERE
Q = int(input())
A = []
for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        A.append(int(query[1]))
    else:
        k = int(query[1])
        print(A[-k])