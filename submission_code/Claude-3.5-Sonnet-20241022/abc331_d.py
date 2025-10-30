def count_black_in_range(pattern, n, r1, c1, r2, c2):
    # Count black squares in one complete pattern block
    def count_black_in_block(r_start, r_end, c_start, c_end):
        count = 0
        for i in range(r_start, r_end):
            for j in range(c_start, c_end):
                if pattern[i % n][j % n] == 'B':
                    count += 1
        return count
    
    # Get complete blocks and remaining cells
    complete_rows = (r2 - r1 + 1) // n
    complete_cols = (c2 - c1 + 1) // n
    
    rem_rows = (r2 - r1 + 1) % n
    rem_cols = (c2 - c1 + 1) % n
    
    # Count black squares in complete pattern blocks
    blacks_per_block = count_black_in_block(r1 % n, (r1 % n) + n, c1 % n, (c1 % n) + n)
    total = blacks_per_block * complete_rows * complete_cols
    
    # Add remaining rows (full width)
    if rem_rows > 0:
        for i in range(r1 + complete_rows * n, r2 + 1):
            for j in range(c1, c1 + complete_cols * n):
                if pattern[i % n][j % n] == 'B':
                    total += 1
                    
    # Add remaining columns (full height including remaining rows)
    if rem_cols > 0:
        for i in range(r1, r2 + 1):
            for j in range(c1 + complete_cols * n, c2 + 1):
                if pattern[i % n][j % n] == 'B':
                    total += 1
                    
    return total

# Read input
N, Q = map(int, input().split())
pattern = [input() for _ in range(N)]

# Process queries
for _ in range(Q):
    A, B, C, D = map(int, input().split())
    result = count_black_in_range(pattern, N, A, B, C, D)
    print(result)