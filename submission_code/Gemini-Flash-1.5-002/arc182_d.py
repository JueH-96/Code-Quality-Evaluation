# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

ans = 0
for i in range(n):
    diff = abs(a[i] - b[i])
    ans += min(diff, m - diff)

print(ans)