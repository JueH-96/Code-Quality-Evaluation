# YOUR CODE HERE

import sys

def solve():
    N = int(input())
    H = list(map(int, input().split()))
    T = 0
    while True:
        for i in range(N):
            if H[i] > 0:
                T += 1
                if T % 3 == 0:
                    H[i] -= 3
                else:
                    H[i] -= 1
            if H[i] <= 0:
                H[i] = 0
        if all(h <= 0 for h in H):
            break
    print(T)

solve()