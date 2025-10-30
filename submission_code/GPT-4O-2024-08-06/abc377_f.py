# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    attacked_rows = set()
    attacked_cols = set()
    attacked_diagonals = set()
    attacked_anti_diagonals = set()
    
    index = 2
    for _ in range(M):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        
        attacked_rows.add(a)
        attacked_cols.add(b)
        attacked_diagonals.add(a + b)
        attacked_anti_diagonals.add(a - b)
    
    # Total number of squares
    total_squares = N * N
    
    # Number of attacked squares
    attacked_by_rows = len(attacked_rows) * N
    attacked_by_cols = len(attacked_cols) * N
    attacked_by_diagonals = len(attacked_diagonals)
    attacked_by_anti_diagonals = len(attacked_anti_diagonals)
    
    # Overlapping squares in rows and columns
    overlap_rows_cols = len(attacked_rows) * len(attacked_cols)
    
    # Total attacked squares
    total_attacked = (attacked_by_rows + attacked_by_cols - overlap_rows_cols +
                      attacked_by_diagonals + attacked_by_anti_diagonals)
    
    # Safe squares
    safe_squares = total_squares - total_attacked
    
    print(safe_squares)

if __name__ == "__main__":
    main()