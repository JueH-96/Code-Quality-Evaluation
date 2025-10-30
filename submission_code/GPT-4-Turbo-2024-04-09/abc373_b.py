def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Create a dictionary to map each character to its position in the string S
    position = {char: idx for idx, char in enumerate(S)}
    
    # We need to type "ABCDEFGHIJKLMNOPQRSTUVWXYZ" in this order
    order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Start at the position of 'A'
    current_pos = position['A']
    total_distance = 0
    
    # Calculate the total distance traveled
    for char in order[1:]:  # Start from the second character since we are already at 'A'
        next_pos = position[char]
        total_distance += abs(next_pos - current_pos)
        current_pos = next_pos
    
    # Output the total distance
    print(total_distance)

if __name__ == "__main__":
    main()