# YOUR CODE HERE
import sys
from collections import Counter

A, B, C, D = map(int, sys.stdin.read().split())

numbers = [A, B, C, D]

found = False
for E in range(1, 14):
    all_numbers = numbers + [E]
    counts = Counter(all_numbers).values()
    if sorted(counts) == [2, 3]:
        print("Yes")
        found = True
        break
if not found:
    print("No")