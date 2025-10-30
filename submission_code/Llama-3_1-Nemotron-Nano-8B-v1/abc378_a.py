from collections import Counter

a = list(map(int, input().split()))
count = Counter(a)
total = sum(v // 2 for v in count.values())
print(total)