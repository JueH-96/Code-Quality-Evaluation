# YOUR CODE HERE
from collections import Counter

a, b, c, d = map(int, input().split())
cards = [a, b, c, d]
counts = Counter(cards)

if len(counts) == 4:
    print("No")
elif len(counts) == 3:
    print("Yes")
elif len(counts) == 2:
    values = list(counts.values())
    if 3 in values or 2 in values:
        print("Yes")
    else:
        print("No")
else:
    print("No")