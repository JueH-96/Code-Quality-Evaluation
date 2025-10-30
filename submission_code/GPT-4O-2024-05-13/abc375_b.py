import sys
import math

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    points = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]
    
    total_cost = 0.0
    
    # Start from the origin
    prev_x, prev_y = 0, 0
    
    for x, y in points:
        total_cost += math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
        prev_x, prev_y = x, y
    
    # Return to the origin
    total_cost += math.sqrt(prev_x ** 2 + prev_y ** 2)
    
    print(f"{total_cost:.15f}")

if __name__ == "__main__":
    main()