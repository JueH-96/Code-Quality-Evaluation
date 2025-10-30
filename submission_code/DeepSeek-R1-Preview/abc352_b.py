# Read the input
S = input().strip()
T = input().strip()

# Initialize variables
i = 0  # pointer for S
positions = []

# Iterate through T to find the positions
for j in range(len(T)):
    if i < len(S) and T[j] == S[i]:
        positions.append(j + 1)  # convert to 1-based index
        i += 1

# Print the result
print(' '.join(map(str, positions)))