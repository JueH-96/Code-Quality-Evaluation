# YOUR CODE HERE

import sys

def solve():
    N = int(sys.stdin.readline().strip())
    S = sys.stdin.readline().strip()

    count = 0
    for i in range(N):
        if S[i] in ['A', 'B', 'C']:
            count += 1
        else:
            break

    print(count)

solve()