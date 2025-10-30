# YOUR CODE HERE
def min_toll(S_x, S_y, T_x, T_y):
    # Calculate the Manhattan distance between the start and target points
    manhattan_distance = abs(S_x - T_x) + abs(S_y - T_y)
    
    # If both points are in the same type of tile (both even or both odd), we need to pay toll only when crossing tiles
    if (S_x + S_y) % 2 == (T_x + T_y) % 2:
        return manhattan_distance // 2
    else:
        # If they are in different types of tiles, we need to pay an additional toll for the last step
        return manhattan_distance // 2 + 1

# Read input
import sys
input = sys.stdin.read
data = input().split()
S_x, S_y, T_x, T_y = map(int, data)

# Calculate and print the minimum toll
print(min_toll(S_x, S_y, T_x, T_y))