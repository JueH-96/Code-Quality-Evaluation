# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))

ans = 0
current = 0
for i in range(n):
    if current + a[i] > k:
        ans += 1
        current = 0
    current += a[i]

if current > 0:
    ans += 1

print(ans)