from collections import Counter

s = input().strip()
count = Counter(s)
max_freq = max(count.values())
candidates = [char for char, freq in count.items() if freq == max_freq]
candidates.sort()
print(candidates[0])