# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    T = int(sys.stdin.readline())
    for _ in range(T):
        N = int(sys.stdin.readline())
        P = list(map(int, sys.stdin.readline().split()))
        is_sorted = True
        for i in range(N-1):
            if P[i] > P[i+1]:
                is_sorted = False
                break
        if is_sorted:
            print(0)
            continue
        any_fixed = False
        for idx, val in enumerate(P):
            if val == idx +1:
                any_fixed = True
                break
        if any_fixed:
            print(1)
        else:
            print(2)

threading.Thread(target=main).start()