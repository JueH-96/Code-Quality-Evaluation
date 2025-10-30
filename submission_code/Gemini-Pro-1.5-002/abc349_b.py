# YOUR CODE HERE
from collections import Counter

s = input()
counts = Counter(s)
freq_counts = Counter(counts.values())

for count in freq_counts.values():
    if count != 0 and count != 2:
        print("No")
        exit()

print("Yes")