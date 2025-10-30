import sys
from collections import Counter

# Read the input
a, b, c, d = map(int, sys.stdin.readline().split())
cards = [a, b, c, d]

# Count the frequency of each card
count = Counter(cards)
freq = sorted(count.values(), reverse=True)

# Check if the frequencies can form a Full House by adding one card
if freq == [3, 1] or freq == [2, 2]:
    print("Yes")
else:
    print("No")