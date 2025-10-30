# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    L_list = []
    R_list = []
    min_R_at_L = [M+1] * (M+2)  # index from 1 to M

    for _ in range(N):
        L_i, R_i = map(int, sys.stdin.readline().split())
        L_list.append(L_i)
        R_list.append(R_i)
        if min_R_at_L[L_i] > R_i:
            min_R_at_L[L_i] = R_i

    minR = [M+1] * (M+2)
    for l in range(M, 0, -1):
        minR[l] = min(minR[l+1], min_R_at_L[l])

    total = 0
    for l in range(1, M+1):
        limit = min(minR[l]-1, M)
        cnt_l = max(0, limit - l +1)
        total += cnt_l
    print(total)

threading.Thread(target=main,).start()