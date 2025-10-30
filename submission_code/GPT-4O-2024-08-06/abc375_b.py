import sys
import math

def calculate_total_cost(N, points):
    # Start at the origin
    current_x, current_y = 0, 0
    total_cost = 0.0
    
    # Visit each point in order
    for x, y in points:
        # Calculate the cost to move from the current point to the next point
        total_cost += math.sqrt((current_x - x) ** 2 + (current_y - y) ** 2)
        # Update the current position
        current_x, current_y = x, y
    
    # Return to the origin
    total_cost += math.sqrt(current_x ** 2 + current_y ** 2)
    
    return total_cost

def main():
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    points = []
    
    index = 1
    for _ in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        points.append((x, y))
        index += 2
    
    result = calculate_total_cost(N, points)
    print(f"{result:.20f}")

if __name__ == "__main__":
    main()