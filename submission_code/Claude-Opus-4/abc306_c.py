# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Dictionary to store positions of each number
positions = {}
for i in range(3 * n):
    num = a[i]
    if num not in positions:
        positions[num] = []
    positions[num].append(i + 1)  # 1-indexed positions

# Create list of (number, middle_position) pairs
f_values = []
for i in range(1, n + 1):
    middle_pos = positions[i][1]  # The second occurrence is the middle one
    f_values.append((i, middle_pos))

# Sort by middle position
f_values.sort(key=lambda x: x[1])

# Extract the sorted numbers
result = [x[0] for x in f_values]

# Print the result
print(' '.join(map(str, result)))