# YOUR CODE HERE
from collections import Counter

S = input().strip()
counter = Counter(S)
max_count = max(counter.values())
most_frequent_chars = [char for char, count in counter.items() if count == max_count]
most_frequent_char = min(most_frequent_chars)
print(most_frequent_char)