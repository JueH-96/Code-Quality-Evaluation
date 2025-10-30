# YOUR CODE HERE
from collections import Counter

s = input().strip()
count = Counter(s)
unique_char = next(char for char, cnt in count.items() if cnt == 1)
print(s.index(unique_char) + 1)