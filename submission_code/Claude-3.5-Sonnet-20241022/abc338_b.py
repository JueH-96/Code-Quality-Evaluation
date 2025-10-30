from collections import Counter

S = input()
freq = Counter(S)
max_freq = max(freq.values())
most_common = [c for c in freq if freq[c] == max_freq]
print(min(most_common))