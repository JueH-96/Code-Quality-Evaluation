# YOUR CODE HERE
import sys
import math
import threading

def main():
    import sys
    import math
    import itertools
    sys.setrecursionlimit(1 << 25)
    N, S, T = map(int, sys.stdin.readline().split())
    N = int(N)
    S = float(S)
    T = float(T)
    A = []
    for _ in range(N):
        A_i, B_i, C_i, D_i = map(int, sys.stdin.readline().split())
        A.append(((A_i, B_i), (C_i, D_i)))

    total_positions = 2*N + 1  # 2 endpoints per segment + start position
    positions = []
    for i in range(N):
        positions.append(A[i][0])  # u_i
        positions.append(A[i][1])  # v_i
    positions.append((0, 0))  # Index 2*N

    # Precompute distances between positions
    dist = [[0.0]*total_positions for _ in range(total_positions)]
    for i in range(total_positions):
        x1, y1 = positions[i]
        for j in range(total_positions):
            x2, y2 = positions[j]
            dx = x1 - x2
            dy = y1 - y2
            dist[i][j] = math.hypot(dx, dy)

    # Precompute length of each segment
    length = []
    for i in range(N):
        u_i = 2*i
        v_i = 2*i + 1
        length_i = dist[u_i][v_i]
        length.append(length_i)

    full_mask = (1 << N) - 1
    dp = [[float('inf')] * total_positions for _ in range(1 << N)]
    dp[0][2*N] = 0.0  # Start from position 2*N (which is (0,0))

    for mask in range(1 << N):
        for u in range(total_positions):
            if dp[mask][u] < float('inf'):
                for i in range(N):
                    if not (mask & (1 << i)):
                        # For segment i, endpoints e_i (2*i) and (2*i+1)
                        for e_start in [2*i, 2*i+1]:
                            e_other = 2*i + 1 if e_start == 2*i else 2*i
                            # Move from current position u to e_start at speed S
                            c_move = dist[u][e_start] / S
                            # Time to print segment i
                            c_print = length[i] / T
                            c_total = dp[mask][u] + c_move + c_print
                            new_mask = mask | (1 << i)
                            if dp[new_mask][e_other] > c_total:
                                dp[new_mask][e_other] = c_total

    ans = min(dp[full_mask][v] for v in range(total_positions))
    print("%.20f" % ans)

threading.Thread(target=main).start()