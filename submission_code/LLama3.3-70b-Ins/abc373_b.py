def calculate_distance(S):
    """
    Calculate the minimal possible total traveled distance of the finger from pressing the key for A to pressing the key for Z.

    Args:
    S (str): A string representing the arrangement of the keyboard, which is a permutation of ABCDEFGHIJKLMNOPQRSTUVWXYZ.

    Returns:
    int: The minimal possible total traveled distance of the finger.
    """
    # Initialize the total distance to 0
    total_distance = 0
    
    # Initialize the current position to the position of 'A'
    current_position = S.index('A') + 1
    
    # Iterate over the characters from 'B' to 'Z'
    for char in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
        # Calculate the position of the current character
        next_position = S.index(char) + 1
        
        # Update the total distance
        total_distance += abs(next_position - current_position)
        
        # Update the current position
        current_position = next_position
    
    # Return the total distance
    return total_distance

# Read the input from stdin
S = input()

# Calculate and print the total distance
print(calculate_distance(S))