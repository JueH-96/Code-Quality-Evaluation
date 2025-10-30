def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    strings = data[1:1+n]
    
    # Find the maximum length M among the strings.
    M = max(len(s) for s in strings)
    
    # Reverse the input strings.
    rev_strings = strings[::-1]
    
    # Build each vertical string T_j for columns 0 to M-1.
    output_lines = []
    for col in range(M):
        col_chars = []
        for row in range(n):
            if col < len(rev_strings[row]):
                col_chars.append(rev_strings[row][col])
            else:
                col_chars.append('*')
        # Create the vertical string from top to bottom.
        vertical_string = "".join(col_chars)
        # Remove trailing '*' to meet the condition that T_i does not end with '*'.
        vertical_string = vertical_string.rstrip('*')
        output_lines.append(vertical_string)
    
    sys.stdout.write("
".join(output_lines))

if __name__ == '__main__':
    main()