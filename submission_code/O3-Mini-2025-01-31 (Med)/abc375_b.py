import sys
import math

def main():
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    n = int(input_data[0])
    coords = list(map(int, input_data[1:]))
    
    total_distance = 0.0
    # starting position at origin
    prev_x, prev_y = 0, 0
    
    # Iterate through each pair representing a point.
    for i in range(n):
        x = coords[2 * i]
        y = coords[2 * i + 1]
        dx = x - prev_x
        dy = y - prev_y
        total_distance += math.sqrt(dx * dx + dy * dy)
        prev_x, prev_y = x, y
        
    # Return to origin
    total_distance += math.sqrt(prev_x * prev_x + prev_y * prev_y)
    print(total_distance)

if __name__ == '__main__':
    main()