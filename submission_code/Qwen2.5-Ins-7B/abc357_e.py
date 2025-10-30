# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
cnt = [0] * n
for i in range(n):
    j = i
    while cnt[j] == 0:
        cnt[j] = i + 1
        j = a[j] - 1
res = 0
for i in range(n):
    res += n - cnt[i]
print(res)