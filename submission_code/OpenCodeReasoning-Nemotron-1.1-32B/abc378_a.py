from collections import Counter

cnt = Counter(map(int, input().split()))
print(sum(v // 2 for v in cnt.values()))