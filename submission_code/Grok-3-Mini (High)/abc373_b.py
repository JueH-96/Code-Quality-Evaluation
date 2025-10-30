# Read the input string from stdin
S = input().strip()

# Create a dictionary mapping each character to its 0-based index in S
pos_dict = {char: idx for idx, char in enumerate(S)}

# Define the alphabet string for the typing order
alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Create a list of positions in the order of typing (A to Z)
pos_list = [pos_dict[char] for char in alpha]

# Calculate the total distance by summing absolute differences between consecutive positions
total_dist = sum(abs(pos_list[i] - pos_list[i-1]) for i in range(1, 26))

# Print the total distance
print(total_dist)