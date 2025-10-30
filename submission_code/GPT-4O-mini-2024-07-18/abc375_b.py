import sys
import math

def total_cost(N, points):
    total_distance = 0.0
    
    # Start from the origin (0, 0)
    current_x, current_y = 0, 0
    
    for x, y in points:
        # Calculate the distance from current position to the next point
        distance = math.sqrt((current_x - x) ** 2 + (current_y - y) ** 2)
        total_distance += distance
        
        # Move to the next point
        current_x, current_y = x, y
    
    # Return to the origin
    distance_to_origin = math.sqrt(current_x ** 2 + current_y ** 2)
    total_distance += distance_to_origin
    
    return total_distance

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    points = [tuple(map(int, line.split())) for line in data[1:N+1]]
    
    result = total_cost(N, points)
    print(f"{result:.20f}")

if __name__ == "__main__":
    main()