from collections import Counter

colors = list(map(int, input().split()))
count = Counter(colors)
result = sum(freq // 2 for freq in count.values())
print(result)