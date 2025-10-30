# Read input strings S and T
S = input().strip()
T = input().strip()

# List to store the positions of correctly typed characters
positions = []

# Pointer for S, starting at the first character
i = 0
# Iterate through T with positions starting from 1
for j in range(1, len(T) + 1):
    if i < len(S) and T[j - 1] == S[i]:
        positions.append(j)
        i += 1
    # If all characters in S are matched, break early
    if i == len(S):
        break

# Print the positions separated by spaces
print(' '.join(map(str, positions)))