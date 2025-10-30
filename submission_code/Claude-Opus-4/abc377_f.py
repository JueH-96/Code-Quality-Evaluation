# YOUR CODE HERE
N, M = map(int, input().split())

# Read existing pieces
pieces = []
for _ in range(M):
    a, b = map(int, input().split())
    pieces.append((a, b))

# Track which rows, columns, and diagonals are occupied
occupied_rows = set()
occupied_cols = set()
occupied_diag_sum = set()  # i+j diagonals
occupied_diag_diff = set()  # i-j diagonals
occupied_squares = set()

for a, b in pieces:
    occupied_rows.add(a)
    occupied_cols.add(b)
    occupied_diag_sum.add(a + b)
    occupied_diag_diff.add(a - b)
    occupied_squares.add((a, b))

# Count total squares that are attacked or occupied
# We need to be careful about overlaps

# First, let's count all squares that are safe
total_squares = N * N

# Count squares that are attacked
attacked_squares = 0

# For each row that has a piece, all N squares in that row are attacked
for row in occupied_rows:
    attacked_squares += N

# For each column that has a piece, all N squares in that column are attacked
# But we need to subtract the overlaps with rows
for col in occupied_cols:
    attacked_squares += N
    # Subtract overlaps with occupied rows
    for row in occupied_rows:
        attacked_squares -= 1

# For each diagonal (i+j = constant), count squares on that diagonal
# But subtract overlaps with rows and columns
for diag_sum in occupied_diag_sum:
    # For diagonal where i+j = diag_sum
    # Valid squares are (i, diag_sum-i) where 1 <= i <= N and 1 <= diag_sum-i <= N
    # This means max(1, diag_sum-N) <= i <= min(N, diag_sum-1)
    min_i = max(1, diag_sum - N)
    max_i = min(N, diag_sum - 1)
    if min_i <= max_i:
        count = max_i - min_i + 1
        attacked_squares += count
        
        # Subtract overlaps with rows
        for row in occupied_rows:
            if min_i <= row <= max_i:
                attacked_squares -= 1
        
        # Subtract overlaps with columns
        for col in occupied_cols:
            i = diag_sum - col
            if min_i <= i <= max_i and i not in occupied_rows:
                attacked_squares -= 1

# For each diagonal (i-j = constant), count squares on that diagonal
# But subtract overlaps with rows, columns, and previous diagonals
for diag_diff in occupied_diag_diff:
    # For diagonal where i-j = diag_diff
    # Valid squares are (i, i-diag_diff) where 1 <= i <= N and 1 <= i-diag_diff <= N
    # This means max(1, 1+diag_diff) <= i <= min(N, N+diag_diff)
    min_i = max(1, 1 + diag_diff)
    max_i = min(N, N + diag_diff)
    if min_i <= max_i:
        count = max_i - min_i + 1
        attacked_squares += count
        
        # Subtract overlaps with rows
        for row in occupied_rows:
            if min_i <= row <= max_i:
                attacked_squares -= 1
        
        # Subtract overlaps with columns
        for col in occupied_cols:
            i = col + diag_diff
            if min_i <= i <= max_i and i not in occupied_rows:
                attacked_squares -= 1
        
        # Subtract overlaps with i+j diagonals
        for diag_sum in occupied_diag_sum:
            # Find intersection of i+j=diag_sum and i-j=diag_diff
            # This gives i = (diag_sum + diag_diff) / 2
            if (diag_sum + diag_diff) % 2 == 0:
                i = (diag_sum + diag_diff) // 2
                j = i - diag_diff
                if min_i <= i <= max_i and 1 <= j <= N:
                    if i not in occupied_rows and j not in occupied_cols:
                        attacked_squares -= 1

# The answer is total squares minus attacked squares
safe_squares = total_squares - attacked_squares
print(safe_squares)