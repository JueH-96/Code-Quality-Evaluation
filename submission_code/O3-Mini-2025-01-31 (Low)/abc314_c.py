def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    index = 0
    n = int(input_data[index]); index += 1
    m = int(input_data[index]); index += 1
    s = list(input_data[index]); index += 1
    colors = list(map(int, input_data[index:index+n])); index += n

    # Build mapping from color number to list of indices (0-indexed)
    color_positions = [[] for _ in range(m+1)]
    for pos in range(n):
        c = colors[pos]
        color_positions[c].append(pos)

    # For each color from 1 to m, perform the right circular shift on s at those indices
    for color in range(1, m+1):
        positions = color_positions[color]
        if len(positions) > 1:
            # Save the last character that will become first in these positions
            last_char = s[positions[-1]]
            # For each index from last to first, shift the previous one to the next slot
            # We can use slicing: assign s[positions[0]] becomes last_char, then for j in range(k-1,0,-1), s[positions[j]]= s[positions[j-1]]
            # Alternatively, use a temporary list.
            # Let's use the temporary list method.
            temp = [s[p] for p in positions]
            # Perform right circular shift on temp: last element moves to front
            temp = [temp[-1]] + temp[:-1]
            # Put back the shifted characters into s list
            for idx, pos in enumerate(positions):
                s[pos] = temp[idx]
        # If there's only one element in positions, the shift doesn't change the character.
    
    # Output final string
    sys.stdout.write(''.join(s))
    
if __name__ == '__main__':
    main()