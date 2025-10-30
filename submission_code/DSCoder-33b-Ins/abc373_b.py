# YOUR CODE HERE

S = input()

# Create a dictionary to store the position of each character in the string
char_pos = {char: pos for pos, char in enumerate(S, 1)}

# Calculate the total distance from 'A' to 'Z'
total_distance = abs(char_pos['A'] - char_pos['Z'])

print(total_distance)