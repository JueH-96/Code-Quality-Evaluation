# Read the input string S from stdin
S = input().strip()

# Initialize the current position at 'A' (which is 0 in 0-indexed)
current_position = S.index('A')
total_distance = 0

# Iterate over the alphabet in order
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    # Find the target position of the current character
    target_position = S.index(char)
    # Calculate the distance to the target position and add it to the total distance
    total_distance += abs(target_position - current_position)
    # Update the current position
    current_position = target_position

# Print the total distance
print(total_distance)