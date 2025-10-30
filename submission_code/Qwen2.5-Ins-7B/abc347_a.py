# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))
result = [a[i] // k for i in range(n) if a[i] % k == 0]
result.sort()
print(*result)