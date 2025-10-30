def main():
    import sys
    S = sys.stdin.read().strip()
    
    # Create a mapping from character to its position
    pos = {char: idx for idx, char in enumerate(S, start=1)}
    
    # Initialize total distance
    total_distance = 0
    
    # Start from position of 'A'
    current_pos = pos['A']
    
    # Iterate from 'B' to 'Z'
    for char in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
        next_pos = pos[char]
        distance = abs(next_pos - current_pos)
        total_distance += distance
        current_pos = next_pos
    
    # Print the total distance
    print(total_distance)

if __name__ == "__main__":
    main()