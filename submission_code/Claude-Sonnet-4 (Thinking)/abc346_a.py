# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

result = []
for i in range(N-1):
    result.append(A[i] * A[i+1])

print(' '.join(map(str, result)))