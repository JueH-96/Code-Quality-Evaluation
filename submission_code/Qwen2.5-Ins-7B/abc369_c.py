# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
ans = 0
for i in range(n):
    d = 0
    for j in range(i, n):
        if j > i:
            d = a[j] - a[j-1]
        if a[j] - a[i] == j - i * d:
            ans += 1
        else:
            break
print(ans)