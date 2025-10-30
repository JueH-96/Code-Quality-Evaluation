# YOUR CODE HERE
import sys
from itertools import groupby

N = int(sys.stdin.readline())
S = sys.stdin.readline().strip()

groups = [len(list(g)) for k, g in groupby(S)]
print(sum((x*(x+1))//2 for x in groups))