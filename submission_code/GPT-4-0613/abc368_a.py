N, K = map(int, input().split())
A = list(map(int, input().split()))
B = A[-K:] + A[:-K]
print(' '.join(map(str, B)))