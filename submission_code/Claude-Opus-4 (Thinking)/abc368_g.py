# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
Q = int(input())

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        i, x = query[1] - 1, query[2]
        A[i] = x
    elif query[0] == 2:
        i, x = query[1] - 1, query[2]
        B[i] = x
    else:  # query[0] == 3
        l, r = query[1] - 1, query[2] - 1
        v = 0
        for i in range(l, r + 1):
            v = max(v + A[i], v * B[i])
        print(v)