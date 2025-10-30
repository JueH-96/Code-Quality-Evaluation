def minimum_toll(S_x, S_y, T_x, T_y):
    # Calculate the difference in coordinates
    delta_x = abs(S_x - T_x)
    delta_y = abs(S_y - T_y)

    # Calculate the number of tiles crossed in x and y directions
    toll_x = (delta_x + 1) // 2
    toll_y = (delta_y + 1) // 2

    # The total toll is the sum of the tolls in both directions
    return toll_x + toll_y

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
S_x, S_y = int(data[0]), int(data[1])
T_x, T_y = int(data[2]), int(data[3])

# Calculate and print the minimum toll
print(minimum_toll(S_x, S_y, T_x, T_y))