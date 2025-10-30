# YOUR CODE HERE
import sys
from collections import Counter

input = sys.stdin.read().strip()
char_count = Counter(input)
max_freq = max(char_count.values())
most_frequent_chars = [char for char, count in char_count.items() if count == max_freq]
print(min(most_frequent_chars))