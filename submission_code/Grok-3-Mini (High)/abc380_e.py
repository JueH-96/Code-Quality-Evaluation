import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Initialize color and freq
color = [0] * N
for i in range(N):
    color[i] = i + 1  # color of cell i+1 is i+1, 0-based index

freq = [0] * (N + 1)
for c in range(1, N + 1):
    freq[c] = 1

# Output list to store results of type 2 queries
output = []

# Process Q queries
for _ in range(Q):
    query_type = int(data[index])
    index += 1
    if query_type == 1:
        x = int(data[index])  # x is 1-based
        index += 1
        c = int(data[index])  # c is the new color
        index += 1
        # Find the component containing x
        idx = x - 1  # Convert to 0-based index
        col = color[idx]  # Current color of the cell
        # Find left boundary L_idx
        L_idx = idx
        while L_idx > 0 and color[L_idx - 1] == col:
            L_idx -= 1
        # Find right boundary R_idx
        R_idx = idx
        while R_idx < N - 1 and color[R_idx + 1] == col:
            R_idx += 1
        S = R_idx - L_idx + 1  # Size of the component
        # Decrease frequency of the old color
        freq[col] -= S
        # Set all cells in the component to new color c
        for i in range(L_idx, R_idx + 1):
            color[i] = c
        # Increase frequency of the new color
        freq[c] += S
    elif query_type == 2:
        c = int(data[index])  # c is the color to query
        index += 1
        # Append the count to output list
        output.append(str(freq[c]))

# Print all outputs for type 2 queries
for out in output:
    print(out)