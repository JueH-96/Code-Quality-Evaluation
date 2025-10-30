# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

ops = 0

for i in range(n):
    diff = abs(a[i] - b[i])
    if diff > m // 2:
        diff = m - diff
    ops += diff

print(ops)