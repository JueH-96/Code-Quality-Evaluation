import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    buildings = []
    for _ in range(n):
        x, h = map(int, sys.stdin.readline().split())
        buildings.append((x, h))
    
    if n == 1:
        print(-1)
        return
    
    dq = deque()
    dq.append(buildings[0])
    max_h = -float('inf')
    
    for i in range(1, n):
        xi, hi = buildings[i]
        x0, h0 = buildings[i-1]
        low, high = 0, len(dq) - 1
        best_j = 0
        while low < high:
            mid = (low + high + 1) // 2
            if mid - 1 >= 0:
                j1_x, j1_h = dq[mid-1]
                j2_x, j2_h = dq[mid]
                t1 = (j1_h * xi - j1_x * hi) / (xi - j1_x)
                t2 = (j2_h * xi - j2_x * hi) / (xi - j2_x)
                if t1 < t2:
                    low = mid
                else:
                    high = mid - 1
            else:
                low = mid
        jx, jh = dq[low]
        current_t = (jh * xi - jx * hi) / (xi - jx)
        if current_t > max_h:
            max_h = current_t
        
        # Maintain the convex hull
        while len(dq) >= 2:
            q = dq[-2]
            r = dq[-1]
            p = (xi, hi)
            cross = (r[0] - q[0]) * (p[1] - r[1]) - (r[1] - q[1]) * (p[0] - r[0])
            if cross >= 0:
                dq.pop()
            else:
                break
        dq.append((xi, hi))
    
    if max_h <= 0:
        print(-1)
    else:
        print("{0:.15f}".format(max_h))

if __name__ == "__main__":
    main()