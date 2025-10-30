n = int(input())
a = list(map(int, input().split()))

from collections import defaultdict
cnt, d , ans = [0]*10**4, defaultdict(int), 0
for i in range(n):
    d[a[i]%10**4] += 1
for i in range(n):
    acc = (a[i]-a[i]%10**4)*d[a[i]%10**4]+cnt[a[i]%10**4]
    cnt[a[i]%10**4] += 10**4
    ans += acc
    d[a[i]%10**4] -= 1
print(ans)