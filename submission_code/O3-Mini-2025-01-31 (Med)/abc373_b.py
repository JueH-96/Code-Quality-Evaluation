def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    S = input_data[0]
    
    # Build a dictionary that maps each letter to its coordinate (1-indexed)
    letter_to_position = {letter: idx+1 for idx, letter in enumerate(S)}
    
    total_distance = 0
    # The order to type is the standard alphabetical order
    order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Start at 'A'
    current_pos = letter_to_position['A']
    
    # Iterate from the second letter onward
    for letter in order[1:]:
        next_pos = letter_to_position[letter]
        total_distance += abs(next_pos - current_pos)
        current_pos = next_pos
        
    print(total_distance)

if __name__ == '__main__':
    main()