# YOUR CODE HERE

import sys

def ctz(n):
    count = 0
    while n & 1 == 0:
        n >>= 1
        count += 1
    return count

N = int(sys.stdin.readline())
print(ctz(N))