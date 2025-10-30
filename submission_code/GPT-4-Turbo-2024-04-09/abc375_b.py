import sys
import math

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    points = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N)]
    
    total_cost = 0.0
    current_x, current_y = 0, 0
    
    # Calculate the cost from origin to each point and between points
    for x, y in points:
        total_cost += math.sqrt((current_x - x) ** 2 + (current_y - y) ** 2)
        current_x, current_y = x, y
    
    # Calculate the cost from the last point back to the origin
    total_cost += math.sqrt(current_x ** 2 + current_y ** 2)
    
    print(total_cost)

if __name__ == "__main__":
    main()