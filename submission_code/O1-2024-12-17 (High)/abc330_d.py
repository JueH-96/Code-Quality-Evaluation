def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    grid = data[1:]  # Each element is a string of length N consisting of 'o' or 'x'
    
    # Arrays to count the number of 'o' in each row and each column
    row_count = [0]*N
    col_count = [0]*N
    
    # First pass: count 'o' in each row and column
    for r in range(N):
        row_str = grid[r]
        ccount = 0
        for c in range(N):
            if row_str[c] == 'o':
                ccount += 1
                col_count[c] += 1
        row_count[r] = ccount
    
    # Second pass: for each cell that has 'o', add (row_count[r] - 1) * (col_count[c] - 1)
    answer = 0
    for r in range(N):
        row_str = grid[r]
        rc = row_count[r] - 1
        if rc > 0:
            for c in range(N):
                if row_str[c] == 'o':
                    cc = col_count[c] - 1
                    if cc > 0:
                        answer += rc * cc
    
    print(answer)

# Do not forget to call main() at the end
main()