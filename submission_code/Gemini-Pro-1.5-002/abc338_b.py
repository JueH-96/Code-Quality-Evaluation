# YOUR CODE HERE
from collections import Counter

s = input()
count = Counter(s)
max_freq = 0
result = ''

for char, freq in count.items():
    if freq > max_freq:
        max_freq = freq
        result = char
    elif freq == max_freq:
        if char < result:
            result = char

print(result)