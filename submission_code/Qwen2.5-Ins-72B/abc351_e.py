# YOUR CODE HERE
import sys

def dist(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if (dx + dy) % 2 == 0:
        return (dx + dy) // 2
    return 0

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    points = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]
    
    total_dist = 0
    for i in range(N-1):
        for j in range(i+1, N):
            total_dist += dist(points[i][0], points[i][1], points[j][0], points[j][1])
    
    print(total_dist)

if __name__ == "__main__":
    main()