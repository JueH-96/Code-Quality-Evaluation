from collections import Counter

a = list(map(int, input().split()))
count = Counter(a)
ans = 0
for c in count.values():
    ans += c // 2
print(ans)