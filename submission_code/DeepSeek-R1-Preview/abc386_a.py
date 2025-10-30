import sys
from collections import Counter

nums = list(map(int, sys.stdin.readline().split()))
counts = sorted(Counter(nums).values(), reverse=True)
if len(counts) >= 2 and ((counts[0] == 3 and counts[1] == 1) or (counts[0] == 2 and counts[1] == 2)):
    print("Yes")
else:
    print("No")