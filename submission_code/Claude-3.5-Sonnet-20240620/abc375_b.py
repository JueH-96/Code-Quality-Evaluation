# YOUR CODE HERE
import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    N = int(input())
    points = [(0, 0)]  # Start with the origin
    
    for _ in range(N):
        x, y = map(int, input().split())
        points.append((x, y))
    
    points.append((0, 0))  # End at the origin
    
    total_cost = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        total_cost += distance(x1, y1, x2, y2)
    
    print(f"{total_cost:.20f}")

if __name__ == "__main__":
    main()