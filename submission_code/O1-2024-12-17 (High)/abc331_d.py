def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    # Parse N and Q
    N, Q = map(int, input_data[:2])
    idx = 2
    
    # Read the N lines of pattern
    pattern = input_data[idx:idx+N]
    idx += N
    
    # Build prefix-sum array of black squares in the N×N pattern.
    # black_count_2d[x][y] = number of black squares in the top-left subrectangle [0..x-1]×[0..y-1]
    black_count_2d = [[0]*(N+1) for _ in range(N+1)]
    
    for i in range(N):
        row_str = pattern[i]
        for j in range(N):
            # Convert 'B' => 1, 'W' => 0
            is_black = (row_str[j] == 'B')
            black_count_2d[i+1][j+1] = (black_count_2d[i][j+1]
                                       + black_count_2d[i+1][j]
                                       - black_count_2d[i][j]
                                       + (1 if is_black else 0))
    
    # Total black squares in one full N×N pattern
    black_count_all = black_count_2d[N][N]
    
    # A fast function to count black squares in [0..X-1]×[0..Y-1] in the infinite tiling
    def count_black_in_big_grid(X, Y):
        # Works even if X or Y is 0 (will return 0).
        xfull = X // N
        xrem = X % N
        yfull = Y // N
        yrem = Y % N
        return (xfull * yfull * black_count_all
                + xfull * black_count_2d[N][yrem]
                + yfull * black_count_2d[xrem][N]
                + black_count_2d[xrem][yrem])
    
    # Process queries
    out = []
    # Each query = A B C D
    # We want count of black squares in [A..C]×[B..D].
    # Using 2D prefix principle:
    # answer = F(C+1, D+1) - F(A, D+1) - F(C+1, B) + F(A, B)
    
    for q in range(Q):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        C = int(input_data[idx+2]); D = int(input_data[idx+3])
        idx += 4
        
        ans = ( count_black_in_big_grid(C+1, D+1)
                - count_black_in_big_grid(A, D+1)
                - count_black_in_big_grid(C+1, B)
                + count_black_in_big_grid(A, B) )
        
        out.append(str(ans))
    
    # Print results
    print("
".join(out))

# Do not forget to call main!
main()