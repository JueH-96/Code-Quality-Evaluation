import bisect
from collections import defaultdict

n = int(input())
a = list(map(int, input().split()))
pos_dict = defaultdict(list)
for idx, num in enumerate(a):
    pos_dict[num].append(idx)
for num in pos_dict:
    pos_dict[num].sort()
ans = [0] * n
ans[0] = n  # k=1
for i in range(n):
    for j in range(i + 1, n):
        d = a[j] - a[i]
        current_min = j
        ans[1] += 1  # k=2
        for l in range(3, n + 1):
            required_value = a[j] + (l - 2) * d
            if required_value not in pos_dict:
                break
            required_list = pos_dict[required_value]
            idx = bisect.bisect_right(required_list, current_min)
            count = len(required_list) - idx
            if count <= 0:
                break
            ans[l - 1] += count
            current_min = required_list[idx]
MOD = 998244353
print(' '.join(map(str, [x % MOD for x in ans])))