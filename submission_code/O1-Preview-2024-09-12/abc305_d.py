# YOUR CODE HERE
import sys
import bisect

import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())
    lr = [tuple(map(int, sys.stdin.readline().split())) for _ in range(Q)]

    N = len(A)
    # status[i] is status during interval [A[i], A[i+1])
    status = []
    for i in range(N):
        status.append(i % 2)
    # cumulative sleep time up to time A[i]
    cumulative_sleep_time = [0] * N
    for i in range(1, N):
        delta = A[i] - A[i - 1]
        if status[i - 1] == 1:
            cumulative_sleep_time[i] = cumulative_sleep_time[i - 1] + delta
        else:
            cumulative_sleep_time[i] = cumulative_sleep_time[i - 1]

    for l_i, r_i in lr:
        # sleep up to l_i
        idx_l = bisect.bisect_right(A, l_i) - 1
        if idx_l >= 0:
            sleep_l = cumulative_sleep_time[idx_l]
            if status[idx_l] == 1:
                sleep_l += l_i - A[idx_l]
        else:
            sleep_l = 0

        idx_r = bisect.bisect_right(A, r_i) - 1
        if idx_r >= 0:
            sleep_r = cumulative_sleep_time[idx_r]
            if status[idx_r] == 1:
                sleep_r += r_i - A[idx_r]
        else:
            sleep_r = 0

        total_sleep = sleep_r - sleep_l
        print(total_sleep)

threading.Thread(target=main).start()