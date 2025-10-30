import sys

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    points = []
    for _ in range(N):
        x, y = map(int, input().split())
        points.append((x, y))
    # For each point i, find the farthest point
    for i in range(N):
        x_i, y_i = points[i]
        max_d2 = -1
        best_id = None
        for j in range(N):
            if i == j:
                continue
            x_j, y_j = points[j]
            dx = x_i - x_j
            dy = y_i - y_j
            d2 = dx*dx + dy*dy
            # Update if we found a larger distance, or same distance but smaller ID
            if d2 > max_d2 or (d2 == max_d2 and (best_id is None or j+1 < best_id)):
                max_d2 = d2
                best_id = j + 1
        print(best_id)

if __name__ == "__main__":
    main()