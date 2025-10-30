def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1:]
    
    # Determine maximum length among all strings
    M = max(len(s) for s in S)
    
    # We'll build each of the M output lines T_j
    # Each T_j has up to N columns; column k (0-based) corresponds to S_{N-k}
    result = []
    for j in range(M):
        # Initialize row with '*' in all N positions
        row = ['*'] * N
        # Fill in letters for each S_i if j < len(S_i)
        # i runs from 1 to N; its index in Python is i-1
        # target column in row is N - i
        for i in range(1, N+1):
            si = S[i-1]
            if j < len(si):
                row[N - i] = si[j]
        # Strip trailing '*' so the line doesn't end with '*'
        # (At least one character is letter, because M = max length)
        while row and row[-1] == '*':
            row.pop()
        result.append(''.join(row))
    
    # Print the result
    print('
'.join(result))


if __name__ == "__main__":
    main()