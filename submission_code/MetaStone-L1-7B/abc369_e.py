import sys
import math

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0

    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1

    INF = math.inf
    dist = [[INF] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        dist[i][i] = 0

    for _ in range(M):
        u = int(data[ptr])
        ptr += 1
        v = int(data[ptr])
        ptr += 1
        t = int(data[ptr])
        ptr += 1
        if dist[u][v] > t:
            dist[u][v] = t
            dist[v][u] = t

    for k in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    Q = int(data[ptr])
    ptr += 1

    for _ in range(Q):
        K = int(data[ptr])
        ptr += 1
        B_list = list(map(int, data[ptr:ptr + K]))
        ptr += K

        B = []
        waypoints = set()
        for b in B_list:
            u = int(data[ptr])
            ptr += 1
            v = int(data[ptr])
            ptr += 1
            t = int(data[ptr])
            ptr += 1
            B.append((u, v, t))
            waypoints.add(u)
            waypoints.add(v)
        waypoints.add(1)
        waypoints.add(N)

        waypoints = list(waypoints)
        n = len(waypoints)
        way_to_idx = {w: i for i, w in enumerate(waypoints)}

        dp = [[INF] * n for _ in range(1 << n)]
        dp[0][0] = 0

        for mask in range(1 << n):
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                current_cost = dp[mask][i]
                if current_cost == INF:
                    continue
                for j in range(n):
                    if mask & (1 << j):
                        if dp[mask][j] < current_cost + dist[i][j]:
                            dp[mask][j] = current_cost + dist[i][j]

        min_cost = INF
        full_mask = (1 << n) - 1
        for j in range(n):
            if dp[full_mask][j] < min_cost:
                min_cost = dp[full_mask][j]

        fixed_sum = 0
        for b in B:
            u, v, t = b
            fixed_sum += t

        if min_cost == INF:
            print(-1)
        else:
            total = fixed_sum + min_cost
            print(total)

if __name__ == "__main__":
    main()