from collections import defaultdict
from heapq import heappush, heappop
import math

def main():
    N, M = map(int, input().split())
    trains = []
    G1 = defaultdict(list)
    for _ in range(M):
        l, d, k, c, A, B = map(int, input().split())
        trains.append((A, B, l, d, k, c))
        G1[A].append((B, l, d, k, c))

    # Calculate max time to reach N
    def max_time(N, G):
        q = []
        max_time = [0] * (N + 1)
        for _, _, l, d, k, _ in trains:
            if _ == N:
                q.append((l + k * d, N))
                max_time[N] = l + k * d

        while q:
            t, node = heappop(q)
            if max_time[node] < t: continue

            for nxt, l, d, k, c in G[node]:
                next_t = min(t - c, l + k * d) - d
                next_k = (next_t - l) // d
                if next_k < 0: continue
                nxt_time = l + next_k * d + c
                if max_time[nxt] < nxt_time:
                    max_time[nxt] = nxt_time
                    heappush(q, (nxt_time, nxt))

        return max_time

    # DP
    G2 = defaultdict(list)
    dp1 = max_time(N, G1)
    for A, B, l, d, k, c in trains:
        if dp1[B] > 0:
            G2[B].append((A, l, d, k, c))
            if dp1[A] == 0: dp1[A] = dp1[B]

    dp2 = max_time(N, G2)

    for i in range(1, N):
        if dp1[i] == 0:
            print("Unreachable")
        else:
            print(min(dp1[i], dp2[i]))
main()