# YOUR CODE HERE
from collections import Counter

s = input()
count = Counter(s)
max_count = 0
result = ''
for char, c in count.items():
    if c > max_count:
        max_count = c
        result = char
    elif c == max_count:
        result = min(result, char)
print(result)