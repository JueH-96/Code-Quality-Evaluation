def main():
    import sys
    input = sys.stdin.readline

    # read the permutation string
    S = input().strip()
    
    # Create dictionary mapping from character to its coordinate (1-indexed)
    pos = {ch: i+1 for i, ch in enumerate(S)}
    
    total_distance = 0
    # Start from the position of 'A'
    current_pos = pos['A']
    
    # The sequence to type: A -> B -> ... -> Z
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        target = pos[ch]
        total_distance += abs(target - current_pos)
        current_pos = target
        
    print(total_distance)

if __name__ == '__main__':
    main()