# YOUR CODE HERE
from collections import Counter

S = input().strip()
count = Counter(S)
freq_count = Counter(count.values())

for i in freq_count:
    if freq_count[i] != 0 and freq_count[i] != 2:
        print("No")
        exit()
print("Yes")