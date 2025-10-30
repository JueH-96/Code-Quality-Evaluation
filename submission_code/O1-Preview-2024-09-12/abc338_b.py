S = input()
from collections import Counter
freq = Counter(S)
max_freq = max(freq.values())
most_common_chars = [char for char in freq if freq[char] == max_freq]
print(min(most_common_chars))