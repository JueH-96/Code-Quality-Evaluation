# YOUR CODE HERE
from collections import Counter

S = input().strip()
count = Counter(S)
freq_count = Counter(count.values())

for key in freq_count:
    if freq_count[key] != 0 and freq_count[key] != 2:
        print("No")
        exit()
print("Yes")