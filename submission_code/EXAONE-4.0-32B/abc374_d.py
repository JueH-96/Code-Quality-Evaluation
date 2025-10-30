import math
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        print(0.0)
        return
    
    it = iter(data)
    n = int(next(it))
    S_val = float(next(it))
    T_val = float(next(it))
    segments = []
    for _ in range(n):
        a = int(next(it))
        b = int(next(it))
        c = int(next(it))
        d = int(next(it))
        segments.append((a, b, c, d))
    
    points = [(0.0, 0.0)]
    for a, b, c, d in segments:
        points.append((float(a), float(b)))
        points.append((float(c), float(d)))
    
    M = len(points)
    dist_mat = [[0.0] * M for _ in range(M)]
    for i in range(M):
        x1, y1 = points[i]
        for j in range(M):
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_mat[i][j] = math.sqrt(dx*dx + dy*dy)
    
    seg_info = []
    for i in range(n):
        u_index = 2 * i + 1
        v_index = 2 * i + 2
        a, b, c, d = segments[i]
        length = math.sqrt((a - c)**2 + (b - d)**2)
        seg_info.append((u_index, v_index, length))
    
    num_masks = 1 << n
    dp = [[10**18] * M for _ in range(num_masks)]
    dp[0][0] = 0.0
    
    for mask in range(num_masks):
        for node in range(M):
            if dp[mask][node] >= 10**18:
                continue
            for i in range(n):
                if mask & (1 << i):
                    continue
                u, v, L = seg_info[i]
                time1 = dist_mat[node][u] / S_val + L / T_val
                new_mask = mask | (1 << i)
                if dp[new_mask][v] > dp[mask][node] + time1:
                    dp[new_mask][v] = dp[mask][node] + time1
                time2 = dist_mat[node][v] / S_val + L / T_val
                if dp[new_mask][u] > dp[mask][node] + time2:
                    dp[new_mask][u] = dp[mask][node] + time2
    
    ans = min(dp[num_masks - 1])
    print("{:.20f}".format(ans))

if __name__ == "__main__":
    main()