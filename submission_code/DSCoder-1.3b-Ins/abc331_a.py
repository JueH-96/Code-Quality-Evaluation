# YOUR CODE HERE

M, D = map(int, input().split())
y, m, d = map(int, input().split())

if d < D:
    print(y, m, d+1)
elif m < M:
    print(y+1, m+1, 1)
else:
    print(y+1, 1, 1)