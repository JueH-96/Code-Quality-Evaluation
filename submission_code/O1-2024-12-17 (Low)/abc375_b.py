import math
import sys

def main():
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    coords = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]
    
    # Start at (0,0), accumulate total distance
    prev_x, prev_y = 0, 0
    total_cost = 0.0
    
    # Move through the points in order
    for x, y in coords:
        dx = x - prev_x
        dy = y - prev_y
        total_cost += math.hypot(dx, dy)
        prev_x, prev_y = x, y
    
    # Return to origin
    total_cost += math.hypot(prev_x, prev_y)
    
    print(total_cost)

# Do not forget to call main()
main()