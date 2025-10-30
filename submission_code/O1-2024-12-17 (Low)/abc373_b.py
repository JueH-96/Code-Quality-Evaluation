def main():
    import sys
    S = sys.stdin.readline().strip()
    # Create a dictionary mapping each letter to its 1-based position on the keyboard
    pos = {}
    for i, c in enumerate(S):
        pos[c] = i + 1

    total_distance = 0
    current_position = pos['A']
    
    for char in "BCDEFGHIJKLMNOPQRSTUVWXYZ":
        total_distance += abs(pos[char] - current_position)
        current_position = pos[char]
    
    print(total_distance)

# DO NOT forget to call main()!
if __name__ == "__main__":
    main()