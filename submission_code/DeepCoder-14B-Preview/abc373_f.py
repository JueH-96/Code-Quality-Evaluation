# Read the input string
S = input().strip()

# Create a dictionary to map each character to its position
position = {char: idx + 1 for idx, char in enumerate(S)}

# Generate the sequence of letters from A to Z
sequence = [chr(ord('A') + i) for i in range(26)]

# Calculate the positions of each letter in the sequence
pos_list = [position[char] for char in sequence]

# Compute the total traveled distance
total_distance = 0
for i in range(1, len(pos_list)):
    total_distance += abs(pos_list[i] - pos_list[i-1])

# Print the result
print(total_distance)