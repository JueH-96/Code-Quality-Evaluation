import sys

def main():
    N = int(sys.stdin.readline())
    D = [[0] * (N + 1) for _ in range(N + 1)]  # 1-based indices

    for i in range(1, N):
        parts = list(map(int, sys.stdin.readline().split()))
        for idx in range(len(parts)):
            j = i + 1 + idx
            D[i][j] = parts[idx]

    max_mask = 1 << N
    dp = [-float('inf')] * max_mask
    dp[0] = 0

    for mask in range(max_mask):
        current = dp[mask]
        if current == -float('inf'):
            continue

        available = []
        for v in range(1, N + 1):
            if not (mask & (1 << (v - 1))):
                available.append(v)

        n = len(available)
        for i in range(n):
            u = available[i]
            for j in range(i + 1, n):
                v = available[j]
                new_mask = mask | (1 << (u - 1)) | (1 << (v - 1))
                if dp[new_mask] < current + D[u][v]:
                    dp[new_mask] = current + D[u][v]

    print(max(dp))

if __name__ == "__main__":
    main()