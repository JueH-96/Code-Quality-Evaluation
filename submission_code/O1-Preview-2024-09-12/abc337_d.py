# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    INF = float('inf')
    min_ops = INF

    # Process rows
    for i in range(H):
        p = 0
        while p < W:
            # Skip 'x'
            if grid[i][p] == 'x':
                p += 1
                continue
            # Start of a segment without 'x'
            start = p
            dot_cnt = []
            while p < W and grid[i][p] != 'x':
                dot_cnt.append(1 if grid[i][p] == '.' else 0)
                p += 1
            length = p - start
            if length < K:
                continue
            # Sliding window to find minimal operations
            cnt_dot = sum(dot_cnt[:K])
            min_ops = min(min_ops, cnt_dot)
            for s in range(1, length - K + 1):
                cnt_dot = cnt_dot - dot_cnt[s - 1] + dot_cnt[s + K - 1]
                min_ops = min(min_ops, cnt_dot)

    # Process columns
    for j in range(W):
        p = 0
        while p < H:
            if grid[p][j] == 'x':
                p += 1
                continue
            # Start of a segment without 'x'
            start = p
            dot_cnt = []
            while p < H and grid[p][j] != 'x':
                dot_cnt.append(1 if grid[p][j] == '.' else 0)
                p += 1
            length = p - start
            if length < K:
                continue
            # Sliding window to find minimal operations
            cnt_dot = sum(dot_cnt[:K])
            min_ops = min(min_ops, cnt_dot)
            for s in range(1, length - K + 1):
                cnt_dot = cnt_dot - dot_cnt[s - 1] + dot_cnt[s + K - 1]
                min_ops = min(min_ops, cnt_dot)

    if min_ops == INF:
        print(-1)
    else:
        print(int(min_ops))

if __name__ == "__main__":
    threading.Thread(target=main).start()