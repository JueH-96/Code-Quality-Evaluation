# YOUR CODE HERE

import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, A, B = map(int, sys.stdin.readline().split())
    M = A + B
    D = list(map(int, sys.stdin.readline().split()))
    R = M -1
    for D_i in D:
        r_i = (D_i -1) % M
        if r_i >= A:
            print("No")
            return
        temp_R = A - r_i - 1
        R = min(R, temp_R)
        # If R becomes negative, no need to continue
        if R < 0:
            print("No")
            return
    print("Yes")


threading.Thread(target=main).start()