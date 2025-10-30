# YOUR CODE HERE
def solve():
    n, q = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    prefix_sum = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(n):
            prefix_sum[i+1][j+1] = prefix_sum[i+1][j] + prefix_sum[i][j+1] - prefix_sum[i][j] + (1 if grid[i][j] == 'B' else 0)

    def count_black(a, b, c, d):
        count = 0
        for i in range(a, c + 1):
            for j in range(b, d + 1):
                if grid[i % n][j % n] == 'B':
                    count += 1
        return count
    
    def count_black_fast(a, b, c, d):
        count = 0
        row_count = c - a + 1
        col_count = d - b + 1
        
        full_rows = row_count // n
        full_cols = col_count // n
        
        total_black = prefix_sum[n][n]
        count += full_rows * full_cols * total_black
        
        rem_rows = row_count % n
        rem_cols = col_count % n
        
        start_row = a % n
        start_col = b % n
        
        for i in range(rem_rows):
            for j in range(n):
                if grid[(start_row + i) % n][j] == 'B':
                    count += full_cols
        
        for j in range(rem_cols):
            for i in range(n):
                if grid[i][(start_col + j) % n] == 'B':
                    count += full_rows

        for i in range(rem_rows):
            for j in range(rem_cols):
                if grid[(start_row + i) % n][(start_col + j) % n] == 'B':
                    count += 1
                    
        return count

    for _ in range(q):
        a, b, c, d = map(int, input().split())
        print(count_black_fast(a, b, c, d))

solve()