# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

result = 0
for i in range(N):
    for j in range(i+1, N):
        result += max(A[j] - A[i], 0)

print(result)