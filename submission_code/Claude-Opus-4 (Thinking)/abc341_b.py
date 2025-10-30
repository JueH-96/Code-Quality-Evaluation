# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

for i in range(N-1):
    S, T = map(int, input().split())
    times = A[i] // S
    A[i+1] += times * T

print(A[N-1])