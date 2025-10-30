import sys

# Read all input data
data = sys.stdin.read().split()

# Extract N and M
N = int(data[0])
M = int(data[1])

# Extract the string S
S_str = data[2]

# Extract and convert C list to integers
C_list = [int(x) for x in data[3:3+N]]

# Convert string to list of characters for mutable operations
chars = list(S_str)

# Group indices by color (0-based color index)
color_indices = [[] for _ in range(M)]
for pos in range(N):
    color = C_list[pos] - 1  # Convert to 0-based index
    color_indices[color].append(pos)

# Perform the shift for each color from 1 to M
for col in range(M):
    idx_list = color_indices[col]
    len_group = len(idx_list)
    if len_group > 0:
        # Get the current values at the indices
        vals = [chars[i] for i in idx_list]
        # Perform right circular shift assignment
        chars[idx_list[0]] = vals[-1]
        for j in range(1, len_group):
            chars[idx_list[j]] = vals[j - 1]

# Join the character list back into a string and print
result = ''.join(chars)
print(result)