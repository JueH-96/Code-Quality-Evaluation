# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    C = list(map(int, data[3:3+N]))
    
    # Create a dictionary to map colors to their respective indices
    color_to_indices = {}
    for idx, color in enumerate(C):
        if color not in color_to_indices:
            color_to_indices[color] = []
        color_to_indices[color].append(idx)
    
    # Perform the right circular shift for each color in order
    for color in range(1, M+1):
        if color in color_to_indices:
            indices = color_to_indices[color]
            if len(indices) > 1:
                # Extract the characters at these indices
                chars = [S[i] for i in indices]
                # Perform the right circular shift
                shifted_chars = [chars[-1]] + chars[:-1]
                # Update the string S
                for i, idx in enumerate(indices):
                    S[idx] = shifted_chars[i]
    
    # Convert the list back to a string
    final_S = ''.join(S)
    print(final_S)

if __name__ == "__main__":
    main()