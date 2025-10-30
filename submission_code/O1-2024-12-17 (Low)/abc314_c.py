def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = list(data[2])
    colors = list(map(int, data[3:]))

    # color_positions[i] will hold all positions of color i (1-indexed for colors)
    color_positions = [[] for _ in range(M+1)]
    for i, c in enumerate(colors):
        color_positions[c].append(i)
    
    # Perform the shifts for each color in ascending order
    for i in range(1, M+1):
        positions = color_positions[i]
        if len(positions) > 1:
            # Collect the characters currently at those positions
            temp_chars = [S[pos] for pos in positions]
            # Circular shift to the right by 1
            # The last element in temp_chars moves to the front
            shifted = [temp_chars[-1]] + temp_chars[:-1]
            # Place them back into S
            for idx, pos in enumerate(positions):
                S[pos] = shifted[idx]
    
    # Print the resulting string
    print("".join(S))

# IMPORTANT: Do not forget to call main()
if __name__ == "__main__":
    main()