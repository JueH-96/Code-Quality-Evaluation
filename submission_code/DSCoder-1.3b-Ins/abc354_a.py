# YOUR CODE HERE

import sys

H = int(sys.stdin.readline().strip())

i = 0
while True:
    if H > 2**i:
        print(i)
        break
    i += 1