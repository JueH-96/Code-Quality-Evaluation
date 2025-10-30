# YOUR CODE HERE
n, x, y = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

dp = [0] * (n + 1)
s = 0
for i in range(n):
    s += a[i]
    dp[i + 1] = s

def check(m):
    s = 0
    for i in range(m):
        s += a[i]
        if s > x:
            return False
    for i in range(m, n):
        if dp[i + 1] - dp[m] > x or s + b[i] > y:
            return False
    return True

left, right = 0, n
while left < right:
    mid = (left + right) // 2
    if check(mid):
        right = mid
    else:
        left = mid + 1

print(left)