import math

def sqrt(x):
    return x ** 0.5

def main():
    N = int(input())
    points = []
    for i in range(N):
        x, y = map(int, input().split())
        points.append((x, y))
    
    # Store the Euclidean distances between all pairs of points
    distances = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                distances[i][j] = sqrt(dx * dx + dy * dy)
    
    # For each point, find the farthest point
    for i in range(N):
        # Find the index of the point with the largest distance, excluding the point itself
        farthest_index = max(range(N), key=lambda j: distances[i][j] if j != i else -1)
        print(farthest_index + 1)

if __name__ == "__main__":
    main()