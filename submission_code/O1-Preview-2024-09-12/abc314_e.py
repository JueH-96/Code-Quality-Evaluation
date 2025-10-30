# YOUR CODE HERE
import sys

import threading

def main():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())

    N = int(N)
    M = int(M)

    wheels = []
    for _ in range(N):
        parts = sys.stdin.readline().split()
        Ci = int(parts[0])
        Pi = int(parts[1])
        Si = list(map(int, parts[2:]))
        wheels.append({'C': Ci, 'P': Pi, 'S': Si})

    MMAX = M
    dp = [float('inf')] * (MMAX + 1)
    dp[M] = 0.0  # base case

    for t in range(M - 1, -1, -1):
        dp_t_min = float('inf')
        for wheel in wheels:
            Ci = wheel['C']
            Pi = wheel['P']
            Si = wheel['S']
            temp = Ci
            for s in Si:
                p_s = 1.0 / Pi
                next_t = t + s
                if next_t >= M:
                    dp_next = 0.0
                else:
                    dp_next = dp[next_t]
                temp += p_s * dp_next
            if temp < dp_t_min:
                dp_t_min = temp
        dp[t] = dp_t_min

    print(dp[0])


if __name__ == "__main__":
    threading.Thread(target=main).start()