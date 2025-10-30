# YOUR CODE HERE

import sys

def full_moon_days(N, M, P):
    count = 0
    while M <= N:
        if M <= N:
            count += 1
        M += P
    return count

N, M, P = map(int, sys.stdin.readline().split())
print(full_moon_days(N, M, P))