# YOUR CODE HERE
n = int(input())
h = list(map(int, input().split()))
a = [0] * (n + 1)
ans = [0] * n
ops = 0
for i in range(n):
    ops += h[i] + 1
    a[i + 1] = h[i]
    if i > 0:
        a[i + 1] += a[i]
    ans[i] = ops

print(*ans)