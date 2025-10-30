# YOUR CODE HERE
import sys
import threading

def main():
    import bisect
    import sys

    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    a_list = list(map(int, sys.stdin.readline().split()))
    a_list.sort()
    for _ in range(Q):
        b_j_str, k_j_str = sys.stdin.readline().split()
        b_j = int(b_j_str)
        k_j = int(k_j_str)
        lo = 0
        hi = 2 * 10 ** 8  # Since coordinates go up to 1e8
        while lo < hi:
            mid = (lo + hi) // 2
            l = b_j - mid
            r = b_j + mid
            count = bisect.bisect_right(a_list, r) - bisect.bisect_left(a_list, l)
            if count >= k_j:
                hi = mid
            else:
                lo = mid + 1
        print(lo)

threading.Thread(target=main).start()