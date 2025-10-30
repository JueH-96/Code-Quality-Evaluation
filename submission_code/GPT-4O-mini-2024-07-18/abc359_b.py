def count_colors_with_one_between(N, A):
    count = 0
    positions = {}

    # Store the positions of each color
    for index, color in enumerate(A):
        if color not in positions:
            positions[color] = []
        positions[color].append(index)

    # Check the condition for each color
    for color in range(1, N + 1):
        if color in positions:
            pos1, pos2 = positions[color]
            if abs(pos1 - pos2) == 2:
                count += 1

    return count

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Get the result and print it
result = count_colors_with_one_between(N, A)
print(result)