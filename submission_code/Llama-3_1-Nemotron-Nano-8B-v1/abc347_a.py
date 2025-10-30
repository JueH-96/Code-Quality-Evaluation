# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))
result = [x // k for x in a if x % k == 0]
print(' '.join(map(str, result)))