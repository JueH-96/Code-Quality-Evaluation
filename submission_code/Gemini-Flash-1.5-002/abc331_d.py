# YOUR CODE HERE
N, Q = map(int, input().split())
P = [input() for _ in range(N)]

for _ in range(Q):
    A, B, C, D = map(int, input().split())
    count = 0
    for i in range(A, C + 1):
        for j in range(B, D + 1):
            if P[i % N][j % N] == 'B':
                count += 1
    print(count)