def count_safe_squares(N, pieces):
    occupied_rows = set()
    occupied_cols = set()
    occupied_diags = set()
    occupied_anti_diags = set()
    
    for row, col in pieces:
        occupied_rows.add(row)
        occupied_cols.add(col)
        occupied_diags.add(row + col)
        occupied_anti_diags.add(row - col)
    
    critical_values = [0, N + 1]  # Boundaries
    
    for d in occupied_diags:
        if 2 <= d <= N + 1:
            critical_values.append(d - 1)
        if d - N <= N:
            critical_values.append(d - N)
    
    for a in occupied_anti_diags:
        if a + 1 <= N:
            critical_values.append(a + 1)
        if a + N >= 1:
            critical_values.append(a + N)
    
    critical_values = sorted(list(set(critical_values)))
    
    total_safe_squares = 0
    
    for i in range(len(critical_values) - 1):
        start_row = max(1, critical_values[i] + 1)
        end_row = min(N, critical_values[i + 1])
        
        if start_row > end_row:
            continue
        
        # Count the number of unoccupied rows in the range [start_row, end_row]
        occupied_in_range = sum(1 for row in occupied_rows if start_row <= row <= end_row)
        unoccupied_rows = (end_row - start_row + 1) - occupied_in_range
        
        if unoccupied_rows == 0:
            continue
        
        # Find a sample row in the range that's not occupied
        sample_row = start_row
        while sample_row <= end_row and sample_row in occupied_rows:
            sample_row += 1
        
        if sample_row > end_row:
            continue  # No unoccupied row in the range
        
        unsafe_columns = set()
        
        # Columns that are occupied
        for col in occupied_cols:
            unsafe_columns.add(col)
        
        # Columns j such that sample_row+j is an occupied diagonal
        for diag in occupied_diags:
            j = diag - sample_row
            if 1 <= j <= N:
                unsafe_columns.add(j)
        
        # Columns j such that sample_row-j is an occupied anti-diagonal
        for anti_diag in occupied_anti_diags:
            j = sample_row - anti_diag
            if 1 <= j <= N:
                unsafe_columns.add(j)
        
        safe_columns = N - len(unsafe_columns)
        total_safe_squares += unoccupied_rows * safe_columns
    
    return total_safe_squares

def main():
    N, M = map(int, input().split())
    pieces = []
    for _ in range(M):
        a, b = map(int, input().split())
        pieces.append((a, b))

    print(count_safe_squares(N, pieces))

if __name__ == "__main__":
    main()