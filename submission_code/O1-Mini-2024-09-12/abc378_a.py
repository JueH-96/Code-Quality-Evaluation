from collections import Counter

balls = list(map(int, input().split()))
counts = Counter(balls)
operations = sum(v // 2 for v in counts.values())
print(operations)