import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    instr = [input().split() for _ in range(Q)]
    # distance on ring
    def dist(a, b):
        d = abs(a - b)
        return min(d, N - d)

    INF = 10**18
    # dp maps static-hand-position -> cost
    dp = [INF] * (N + 1)
    active = []

    # Initialize with first instruction
    H1, T1s = instr[0]
    T1 = int(T1s)
    if H1 == 'L':
        # move left from 1 to T1, right stays at 2
        cost = dist(1, T1)
        dp[2] = cost
        active = [2]
    else:
        # move right from 2 to T1, left stays at 1
        cost = dist(2, T1)
        dp[1] = cost
        active = [1]
    prev_H = H1
    prev_t = T1

    for i in range(1, Q):
        H, Ts = instr[i]
        t = int(Ts)
        new_dp = [INF] * (N + 1)
        new_active = []

        if H == 'L':
            # now left is at t, static is right at j_cur
            if prev_H == 'L':
                # previous left at prev_t, static right at j in active
                step_cost = dist(prev_t, t)
                for j in active:
                    if j == t:
                        continue
                    c = dp[j] + step_cost
                    new_dp[j] = c
                    new_active.append(j)
            else:
                # prev_H == 'R'
                # previous right at prev_t, static left at j in active
                # now moving left from j to t, so static right j_cur = prev_t
                best = INF
                for j in active:
                    c = dp[j] + dist(j, t)
                    if c < best:
                        best = c
                jcur = prev_t
                if jcur != t and best < INF:
                    new_dp[jcur] = best
                    new_active = [jcur]

        else:  # H == 'R'
            # now right is at t, static is left at j_cur
            if prev_H == 'R':
                # previous right at prev_t, static left at j in active
                step_cost = dist(prev_t, t)
                for j in active:
                    if j == t:
                        continue
                    c = dp[j] + step_cost
                    new_dp[j] = c
                    new_active.append(j)
            else:
                # prev_H == 'L'
                # previous left at prev_t, static right at j in active
                # now moving right from j to t, static left j_cur = prev_t
                best = INF
                for j in active:
                    c = dp[j] + dist(j, t)
                    if c < best:
                        best = c
                jcur = prev_t
                if jcur != t and best < INF:
                    new_dp[jcur] = best
                    new_active = [jcur]

        # swap in new
        dp = new_dp
        active = new_active
        prev_H = H
        prev_t = t

    # answer is minimal cost over dp at active positions
    ans = INF
    for j in active:
        if dp[j] < ans:
            ans = dp[j]
    print(ans)


if __name__ == "__main__":
    main()