# YOUR CODE HERE
A = list(map(int, input().split()))
result = sum(A[i] * (2 ** i) for i in range(64))
print(result)