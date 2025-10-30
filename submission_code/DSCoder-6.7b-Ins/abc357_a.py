# YOUR CODE HERE

import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    hands = list(map(int, sys.stdin.readline().split()))

    hands.sort()

    count = 0
    for hand in hands:
        if M >= hand:
            M -= hand
            count += 1
        else:
            break

    print(count)

solve()