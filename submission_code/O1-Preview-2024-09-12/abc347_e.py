# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    N, Q = map(int, sys.stdin.readline().split())
    x_list = list(map(int, sys.stdin.readline().split()))
    counts = [0] * (N + 1)
    S_size = [0] * (Q + 1)
    cum_S_size = [0] * (Q + 1)
    toggles = [[] for _ in range(N + 1)]

    for t in range(1, Q + 1):
        xi = x_list[t - 1]
        counts[xi] += 1
        if counts[xi] % 2 == 1:
            # xi is inserted into S
            S_size[t] = S_size[t - 1] + 1
        else:
            # xi is removed from S
            S_size[t] = S_size[t - 1] - 1
        cum_S_size[t] = cum_S_size[t - 1] + S_size[t]
        toggles[xi].append(t)

    A = [0] * N
    for j in range(1, N + 1):
        times = toggles[j]
        intervals = []
        idx = 0
        len_times = len(times)
        while idx + 1 < len_times:
            L = times[idx]
            R = times[idx + 1]
            intervals.append((L, R))
            idx += 2
        if idx < len_times:
            # There's an unmatched toggle, so it's in S till the end
            L = times[idx]
            R = Q + 1
            intervals.append((L, R))
        total = 0
        for L, R in intervals:
            sum_interval = cum_S_size[R - 1] - cum_S_size[L - 1]
            total += sum_interval
        A[j - 1] = total
    print(' '.join(map(str, A)))

threading.Thread(target=main).start()