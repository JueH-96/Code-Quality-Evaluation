def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1:]
    
    # Count how many 'o' in each row and each column
    row_counts = [0] * N
    col_counts = [0] * N
    
    for r in range(N):
        for c in range(N):
            if S[r][c] == 'o':
                row_counts[r] += 1
                col_counts[c] += 1
    
    # For each cell (r, c) with 'o', compute how many valid triples
    # where (r, c) is the unique "pivot" sharing row with one cell
    # and column with another.
    answer = 0
    for r in range(N):
        for c in range(N):
            if S[r][c] == 'o':
                # (row_counts[r] - 1) = number of other 'o' in row r
                # (col_counts[c] - 1) = number of other 'o' in column c
                answer += (row_counts[r] - 1) * (col_counts[c] - 1)
    
    print(answer)

# Do not forget to call main()!
main()