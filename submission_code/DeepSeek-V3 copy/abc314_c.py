def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    C = list(map(int, data[3:3+N]))
    
    # Create a dictionary to map each color to its list of indices
    color_to_indices = {i: [] for i in range(1, M+1)}
    for idx, color in enumerate(C):
        color_to_indices[color].append(idx)
    
    # Perform the right circular shift for each color
    for color in range(1, M+1):
        indices = color_to_indices[color]
        if not indices:
            continue
        # Extract the characters at these indices
        chars = [S[i] for i in indices]
        # Perform the right circular shift
        shifted_chars = [chars[-1]] + chars[:-1]
        # Update the string S
        for i, char in zip(indices, shifted_chars):
            S[i] = char
    
    # Convert the list back to a string and print
    print(''.join(S))

if __name__ == "__main__":
    main()