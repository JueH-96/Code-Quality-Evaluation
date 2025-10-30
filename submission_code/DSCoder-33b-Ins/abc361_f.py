import sys
import math

def solve(n):
    count = 0
    for b in range(2, int(math.log(n, 2)) + 1):
        count += int(math.log(n, b)) - 1
    return count

n = int(sys.stdin.readline())
print(solve(n))