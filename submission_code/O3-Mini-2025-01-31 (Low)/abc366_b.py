def main():
    import sys
    input = sys.stdin.readline
    
    # Read the number of strings and the strings themselves.
    N = int(input().strip())
    strings = [input().strip() for _ in range(N)]
    
    # Determine maximum length M among the given strings.
    M = max(len(s) for s in strings)
    
    # We have a mapping:
    # For each given string S_i (1-indexed), its characters are placed in the vertical column
    # such that the (N - i + 1)-th letter in that column equals S_i[j] (where j is the 1-indexed character in S_i).
    #
    # To facilitate this, we reverse the input strings.
    # Let grid[0] be S_N, grid[1] be S_(N-1), ... grid[N-1] be S_1.
    grid = strings[::-1]
    
    # For each column j (0-indexed), we form a vertical string T_j.
    # Each T_j gets a letter from each row (if exists) or a '*' if that row doesn't have a letter at this column.
    # Trailing '*'s are then stripped for each T_j.
    results = []
    for col in range(M):
        temp = []
        for row in range(N):
            # if the current string (grid[row]) has a character at position 'col', use it.
            if col < len(grid[row]):
                temp.append(grid[row][col])
            else:
                temp.append('*')
        # Join the column characters.
        t_str = "".join(temp)
        # Remove trailing '*' characters.
        t_str = t_str.rstrip('*')
        results.append(t_str)
    
    # Print the vertical strings.
    print("
".join(results))
    
if __name__ == '__main__':
    main()