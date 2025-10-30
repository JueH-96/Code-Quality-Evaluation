import sys
from collections import Counter

l = list(map(int, sys.stdin.readline().strip().split(' ')))
c = Counter(l)
v = [i for i in c.values()]
if (1 in v and 3 in v) or (1 in v and 2 in v) or (2 in v):
    print("Yes")
else:
    print("No")