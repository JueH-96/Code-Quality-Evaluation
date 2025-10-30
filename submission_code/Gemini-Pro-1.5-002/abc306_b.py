# YOUR CODE HERE
a = list(map(int, input().split()))
ans = 0
for i in range(64):
    ans += a[i] * (2**i)
print(ans)