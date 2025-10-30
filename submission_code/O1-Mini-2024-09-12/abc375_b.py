import sys
import math

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    coords = list(map(int, input[1:]))
    
    prev_x, prev_y = 0, 0
    total_cost = 0.0
    
    for i in range(N):
        x = coords[2*i]
        y = coords[2*i + 1]
        dx = x - prev_x
        dy = y - prev_y
        total_cost += math.hypot(dx, dy)
        prev_x, prev_y = x, y
    
    # Return to origin
    total_cost += math.hypot(prev_x, prev_y)
    
    print(total_cost)

if __name__ == "__main__":
    main()