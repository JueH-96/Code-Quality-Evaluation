# YOUR CODE HERE
A = list(map(int, input().split()))
result = 0
for i in range(64):
    result += A[i] * (2 ** i)
print(result)