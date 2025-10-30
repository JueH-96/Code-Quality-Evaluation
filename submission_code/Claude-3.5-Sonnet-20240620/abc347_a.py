# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

result = sorted([a // K for a in A if a % K == 0])
print(*result)