def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    pts = [tuple(map(int, input().split())) for _ in range(N)]

    for i in range(N):
        xi, yi = pts[i]
        best_dist = -1
        best_j = -1
        for j in range(N):
            if i == j:
                continue
            xj, yj = pts[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx*dx + dy*dy
            # update if strictly farther, or same distance but smaller ID
            if d2 > best_dist or (d2 == best_dist and j < best_j):
                best_dist = d2
                best_j = j
        # +1 because IDs are 1-based
        print(best_j + 1)

main()