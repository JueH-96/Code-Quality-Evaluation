# YOUR CODE HERE
N = int(input())
H = list(map(int, input().split()))
A = [0]*(N+1)
for i in range(N):
    if H[i] > A[i]:
        A[i+1] = H[i]
    else:
        A[i+1] = A[i]
print(' '.join(map(str, A[1:])))