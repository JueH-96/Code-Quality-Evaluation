# YOUR CODE HERE
import sys

import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 20)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    H = list(map(int,N_and_rest[1:N+1]))

    from collections import defaultdict

    max_length = 1

    height_positions = defaultdict(list)
    for idx, h in enumerate(H):
        height_positions[h].append(idx)

    for h, positions in height_positions.items():
        P = positions
        M = len(P)
        dp = [dict() for _ in range(M)]
        for j in range(M):
            for i in range(j):
                diff = P[j] - P[i]
                dp_j = dp[j]
                dp_i = dp[i]
                length = dp_i.get(diff, 1) + 1
                dp_j[diff] = length
                if length > max_length:
                    max_length = length

    print(max_length)

threading.Thread(target=main).start()