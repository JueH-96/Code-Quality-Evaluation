# YOUR CODE HERE

import sys

def solve():
    S = sys.stdin.readline().strip()
    suffix = S.split('.')[-1]
    print(suffix)

solve()