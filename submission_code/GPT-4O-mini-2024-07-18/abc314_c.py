def circular_shift(N, M, S, colors):
    # Create a list to hold the characters of S for mutability
    S_list = list(S)
    
    # Create a mapping from color to indices
    color_indices = {}
    for index, color in enumerate(colors):
        if color not in color_indices:
            color_indices[color] = []
        color_indices[color].append(index)
    
    # Perform the right circular shift for each color
    for color in range(1, M + 1):
        if color in color_indices:
            indices = color_indices[color]
            if len(indices) > 1:  # Only shift if there are multiple characters
                # Get the last character to shift to the front
                last_char = S_list[indices[-1]]
                # Shift the characters
                for i in range(len(indices) - 1, 0, -1):
                    S_list[indices[i]] = S_list[indices[i - 1]]
                S_list[indices[0]] = last_char
    
    # Join the list back into a string
    return ''.join(S_list)

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

# First line contains N and M
N, M = map(int, data[0].split())
# Second line contains the string S
S = data[1]
# Third line contains the colors
colors = list(map(int, data[2].split()))

# Get the result and print it
result = circular_shift(N, M, S, colors)
print(result)