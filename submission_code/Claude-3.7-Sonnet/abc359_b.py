# Read the input
N = int(input())
A = list(map(int, input().split()))

# Initialize a dictionary to store positions for each color
color_positions = {}
for j in range(2*N):
    color = A[j]
    if color in color_positions:
        color_positions[color].append(j+1)  # 1-indexed positions
    else:
        color_positions[color] = [j+1]

# Count how many colors have exactly one person between their occurrences
count = 0
for i in range(1, N + 1):
    left, right = color_positions[i]
    if right - left == 2:  # Exactly one person between
        count += 1

print(count)