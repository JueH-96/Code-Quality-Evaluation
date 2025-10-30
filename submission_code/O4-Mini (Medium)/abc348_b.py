def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    pts = [tuple(map(int, input().split())) for _ in range(N)]
    
    # For each point i, find the farthest point (by squared distance).
    res = [0] * N
    for i in range(N):
        xi, yi = pts[i]
        max_dist = -1
        ans_id = None
        for j in range(N):
            if i == j:
                continue
            xj, yj = pts[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx*dx + dy*dy
            # if we find a strictly larger distance, or equal but smaller ID
            if d2 > max_dist or (d2 == max_dist and (j+1) < ans_id):
                max_dist = d2
                ans_id = j + 1
        res[i] = ans_id

    # Output
    print('
'.join(map(str, res)))

if __name__ == "__main__":
    main()