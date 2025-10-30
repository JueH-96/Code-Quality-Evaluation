# YOUR CODE HERE
def minimal_traveled_distance(S):
    # Create a dictionary to map each character to its position in the string S
    position_map = {char: idx for idx, char in enumerate(S)}
    
    # Start at the position of 'A'
    current_position = position_map['A']
    total_distance = 0
    
    # Iterate over each character from 'A' to 'Z'
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        # Calculate the distance to the next character
        next_position = position_map[char]
        total_distance += abs(next_position - current_position)
        # Move the current position to the next character's position
        current_position = next_position
    
    return total_distance

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    result = minimal_traveled_distance(S)
    print(result)