from collections import Counter

s = input().strip()
counts = Counter(s)
max_count = max(counts.values())
candidates = [char for char, cnt in counts.items() if cnt == max_count]
result = min(candidates)
print(result)