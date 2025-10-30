def main():
    import sys
    # Read and clean the input.
    s = sys.stdin.read().strip()
    if not s:
        return
    # Create a dictionary mapping each character to its coordinate (1-indexed).
    coord = {char: idx + 1 for idx, char in enumerate(s)}

    # We'll type ABCDEFGHIJKLMNOPQRSTUVWXYZ in order.
    sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Start from the key for A.
    current_pos = coord['A']
    total_distance = 0
    
    # For each subsequent letter, add the distance from the current position.
    for char in sequence[1:]:
        next_pos = coord[char]
        total_distance += abs(next_pos - current_pos)
        current_pos = next_pos
    
    print(total_distance)

# Call the main function.
if __name__ == '__main__':
    main()