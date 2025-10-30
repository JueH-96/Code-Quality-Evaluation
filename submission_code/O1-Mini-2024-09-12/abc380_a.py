n = input().strip()
from collections import Counter

count = Counter(n)
if count.get('1', 0) == 1 and count.get('2', 0) == 2 and count.get('3', 0) == 3:
    print("Yes")
else:
    print("No")