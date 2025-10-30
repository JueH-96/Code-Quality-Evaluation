# YOUR CODE HERE

N, S, M, L = map(int, input().split())

if N % 12 == 0:
    print(N // 12 * L)
elif N % 8 == 0:
    print(N // 8 * M)
else:
    print((N // 12 * L) + (N % 12 // 8 * M) + (N % 8 * S))