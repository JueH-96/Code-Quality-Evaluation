def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1:]  # The N input strings

    # M is the maximum length among the strings
    M = max(len(x) for x in S)

    # Create a matrix of size M x N, all initialized to None
    # matrix[row][col] in 0-based indexing
    matrix = [[None]*N for _ in range(M)]

    # Fill the matrix so that the i-th string S[i] (0-based) goes into column (N - 1 - i)
    # row index increases with each character of S[i]
    for i in range(N):
        length = len(S[i])
        col = N - 1 - i
        for row in range(length):
            matrix[row][col] = S[i][row]

    # Build each output line T_j (for j in 0-based, meaning T_(j+1) in 1-based)
    for row in range(M):
        # Find the rightmost column where there's a letter
        max_col = -1
        for c in range(N):
            if matrix[row][c] is not None:
                max_col = c

        # If no letters at all in this row, print empty line
        if max_col == -1:
            print("")
            continue

        # Otherwise build the string from column 0..max_col
        # Fill missing positions with '*'
        result = []
        for c in range(max_col + 1):
            if matrix[row][c] is None:
                result.append("*")
            else:
                result.append(matrix[row][c])
        print("".join(result))

# Do not forget to call main()
if __name__ == "__main__":
    main()