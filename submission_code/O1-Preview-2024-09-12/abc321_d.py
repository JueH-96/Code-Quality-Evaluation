# YOUR CODE HERE
import sys
import threading
def main():
    import bisect
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M, P = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    A_sum = sum(A)
    B_sum = sum(B)

    total_s = A_sum * M + B_sum * N

    B.sort()
    B_sum_suffix = [0] * (M +1)
    for i in range(M -1, -1, -1):
        B_sum_suffix[i] = B_sum_suffix[i +1] + B[i]

    total_excess = 0
    for Ai in A:
        idx = bisect.bisect_right(B, P - Ai)
        cnt = M - idx
        if cnt > 0:
            total_Bj = B_sum_suffix[idx]
            total_excess_Ai = cnt * Ai + total_Bj - cnt * P
            total_excess += total_excess_Ai

    total_price = total_s - total_excess
    print(total_price)
threading.Thread(target=main).start()