n, k = map(int, input().split())
A = list(map(int, input().split()))
B = A[-k:] + A[:-k]
print(' '.join(map(str, B)))