# YOUR CODE HERE
import sys

def ctz(N):
    count = 0
    while N % 2 == 0 and N != 0:
        N //= 2
        count += 1
    return count

N = int(input().strip())
print(ctz(N))