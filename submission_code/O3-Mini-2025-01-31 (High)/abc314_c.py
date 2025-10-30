def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # Read N and M
    N = int(input_data[0])
    M = int(input_data[1])
    # Read the string S and convert to a list of characters.
    S = list(input_data[2])
    # Read the colors for each character. They are guaranteed to be M colors in total.
    colors = list(map(int, input_data[3:3+N]))
    
    # Prepare groups: for each color i (1-indexed) maintain a list of indices with that color.
    groups = [[] for _ in range(M+1)]
    for idx, col in enumerate(colors):
        groups[col].append(idx)
        
    # For each color from 1 to M, perform the right circular shift on the characters of S with that color.
    for color in range(1, M+1):
        indices = groups[color]
        if not indices:
            continue
        # Capture the shifted letters for this color group.
        # The right circular shift puts the last letter of the group into the first position.
        new_letters = [S[indices[-1]]] + [S[i] for i in indices[:-1]]
        # Update the string positions with their new characters.
        for pos, letter in zip(indices, new_letters):
            S[pos] = letter
    
    # Write the final string after all operations.
    sys.stdout.write("".join(S))

if __name__ == '__main__':
    main()