# YOUR CODE HERE
import sys

def solve(H):
    i = 0
    while 2**i <= H:
        i += 1
    return i

H = int(sys.stdin.readline())
print(solve(H))