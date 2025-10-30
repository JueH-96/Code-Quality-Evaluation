# YOUR CODE HERE

N, L, R = map(int, input().split())
A = list(map(int, input().split()))

A = A[:(L-1)] + A[R:][::-1] + A[(L-1)+(R-L+1):]

print(*A)