N, M = map(int, input().split())

occupied_rows = set()
occupied_cols = set()
occupied_diag1 = set()  # i+j
occupied_diag2 = set()  # i-j

for _ in range(M):
    a, b = map(int, input().split())
    occupied_rows.add(a)
    occupied_cols.add(b)
    occupied_diag1.add(a + b)
    occupied_diag2.add(a - b)

count = 0

# For each unoccupied row
for i in range(1, N + 1):
    if i in occupied_rows:
        continue
    
    # Count invalid columns for this row
    invalid_cols = set(occupied_cols)
    
    # Add columns that would place piece on occupied diagonal
    for d in occupied_diag1:
        j = d - i
        if 1 <= j <= N:
            invalid_cols.add(j)
    
    # Add columns that would place piece on occupied anti-diagonal
    for d in occupied_diag2:
        j = i - d
        if 1 <= j <= N:
            invalid_cols.add(j)
    
    # Valid columns are total columns minus invalid ones
    count += N - len(invalid_cols)

print(count)