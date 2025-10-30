# YOUR CODE HERE

A, M, L, R = map(int, input().split())

start = max(A // M * M, A)
end = min(R // M * M, R)

if start > end:
    print(0)
else:
    print((end - start) // M + 1)