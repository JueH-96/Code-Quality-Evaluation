# YOUR CODE HERE
import sys
import bisect

import threading
def main():
    import sys

    sys.setrecursionlimit(1 << 25)

    N = int(sys.stdin.readline())
    positions = list(map(int, sys.stdin.readline().split()))
    pops = list(map(int, sys.stdin.readline().split()))

    cum_pops = [0] * N
    cum_pops[0] = pops[0]
    for i in range(1, N):
        cum_pops[i] = cum_pops[i -1] + pops[i]

    Q = int(sys.stdin.readline())
    for _ in range(Q):
        L_str, R_str = sys.stdin.readline().split()
        L = int(L_str)
        R = int(R_str)

        left = bisect.bisect_left(positions, L)
        right = bisect.bisect_right(positions, R)

        if left >= right:
            ans = 0
        else:
            if left > 0:
                ans = cum_pops[right -1] - cum_pops[left -1]
            else:
                ans = cum_pops[right -1]
        print(ans)
threading.Thread(target=main).start()