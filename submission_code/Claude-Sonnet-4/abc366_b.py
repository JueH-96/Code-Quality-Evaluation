# YOUR CODE HERE
n = int(input())
strings = []
for _ in range(n):
    strings.append(input().strip())

# Find maximum length
max_len = max(len(s) for s in strings)

# Create grid - rows are the output lines, columns are the input strings
grid = []
for i in range(max_len):
    row = ['*'] * n
    grid.append(row)

# Fill the grid
for i in range(n):
    s = strings[i]
    col = n - i - 1  # Column position for string i (0-indexed)
    
    for j in range(len(s)):
        grid[j][col] = s[j]

# Convert grid to output strings
result = []
for row in grid:
    # Convert row to string
    line = ''.join(row)
    # Remove trailing '*'
    line = line.rstrip('*')
    result.append(line)

# Print result
for line in result:
    print(line)