import sys
data = sys.stdin.read().split()
N = int(data[0])
A = [int(x) for x in data[1:]]

# Create a list to store positions for each color, index 0 is unused
positions = [[] for _ in range(N + 1)]

# Populate the positions for each color
for pos, color in enumerate(A, start=1):  # pos is 1-based position
    positions[color].append(pos)

# Count how many colors have positions differing by exactly 2
count = 0
for color in range(1, N + 1):
    pos1, pos2 = positions[color][0], positions[color][1]
    if abs(pos1 - pos2) == 2:
        count += 1

# Output the result
print(count)