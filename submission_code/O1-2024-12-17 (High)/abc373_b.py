def main():
    S = input().strip()
    
    # Build a dictionary mapping each character to its position (1-based index)
    pos = {ch: i+1 for i, ch in enumerate(S)}
    
    # We start at 'A'
    current_position = pos['A']
    total_distance = 0
    
    # Calculate distance moving from A -> B -> C -> ... -> Z in order
    for letter in "BCDEFGHIJKLMNOPQRSTUVWXYZ":
        total_distance += abs(pos[letter] - current_position)
        current_position = pos[letter]
    
    print(total_distance)

# Do not remove the call to main()
main()