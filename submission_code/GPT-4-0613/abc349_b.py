# YOUR CODE HERE
from collections import Counter

S = input().strip()
counter = Counter(S)
values = list(counter.values())
counter2 = Counter(values)

for v in counter2.values():
    if v != 0 and v != 2:
        print('No')
        break
else:
    print('Yes')