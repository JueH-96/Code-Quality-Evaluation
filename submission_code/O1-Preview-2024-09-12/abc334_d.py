# YOUR CODE HERE
import sys
import bisect

import threading

def main():
    import sys
    input = sys.stdin.readline

    N, Q = map(int, sys.stdin.readline().split())
    R = list(map(int, sys.stdin.readline().split()))
    R.sort()
    prefix = [0]
    for r in R:
        prefix.append(prefix[-1] + r)
    S = prefix  # prefix sums, 0-indexed

    for _ in range(Q):
        X = int(sys.stdin.readline())
        m = bisect.bisect_right(S, X) -1  # Subtract 1 to get the maximum number of sleighs
        print(m)

threading.Thread(target=main).start()