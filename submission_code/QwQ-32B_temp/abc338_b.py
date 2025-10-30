from collections import Counter

s = input().strip()
count = Counter(s)
max_count = max(count.values())
candidates = [char for char, cnt in count.items() if cnt == max_count]
candidates.sort()
print(candidates[0])