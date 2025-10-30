def find_matrices_with_sums(row_sums, col_sums, n):
    """Find all n×n 0-1 matrices with given row and column sums."""
    if sum(row_sums) != sum(col_sums):
        return []
    
    matrices = []
    
    def backtrack(matrix, i, j):
        if i == n:
            # Verify column sums
            for c in range(n):
                if sum(matrix[r][c] for r in range(n)) != col_sums[c]:
                    return
            matrices.append([row[:] for row in matrix])
            return
        
        if j == n:
            # Verify row sum and move to next row
            if sum(matrix[i]) == row_sums[i]:
                backtrack(matrix, i + 1, 0)
            return
        
        # Calculate constraints
        current_row_sum = sum(matrix[i][:j])
        needed_in_row = row_sums[i] - current_row_sum
        remaining_in_row = n - j
        
        current_col_sum = sum(matrix[r][j] for r in range(i))
        needed_in_col = col_sums[j] - current_col_sum
        remaining_in_col = n - i
        
        # Try placing 0
        if needed_in_row <= remaining_in_row - 1 and needed_in_col <= remaining_in_col - 1:
            matrix[i][j] = 0
            backtrack(matrix, i, j + 1)
        
        # Try placing 1
        if needed_in_row >= 1 and needed_in_col >= 1:
            matrix[i][j] = 1
            backtrack(matrix, i, j + 1)
    
    matrix = [[0] * n for _ in range(n)]
    backtrack(matrix, 0, 0)
    return matrices

def count_fixed_elements(matrices, n):
    """Count the number of fixed elements."""
    if not matrices:
        return -1
    
    fixed_count = 0
    for i in range(n):
        for j in range(n):
            val = matrices[0][i][j]
            if all(m[i][j] == val for m in matrices):
                fixed_count += 1
    return fixed_count

def generate_sum_patterns(n, total):
    """Generate all ways to partition total into n non-negative integers ≤ n."""
    if n == 0:
        return [[]] if total == 0 else []
    
    patterns = []
    for first in range(min(total + 1, n + 1)):
        for rest in generate_sum_patterns(n - 1, total - first):
            patterns.append([first] + rest)
    return patterns

# Read input
n, q = map(int, input().split())

# Find all possible counts of fixed elements
possible_counts = set()
seen_patterns = set()

# Try all possible sum totals
for total in range(n * n + 1):
    # Generate row sum patterns
    for row_sums in generate_sum_patterns(n, total):
        # Generate column sum patterns with same total
        for col_sums in generate_sum_patterns(n, total):
            # Use sorted tuples as canonical form to avoid duplicates
            pattern_key = (tuple(sorted(row_sums)), tuple(sorted(col_sums)))
            if pattern_key in seen_patterns:
                continue
            seen_patterns.add(pattern_key)
            
            # Find all matrices with these sums
            matrices = find_matrices_with_sums(row_sums, col_sums, n)
            if matrices:
                count = count_fixed_elements(matrices, n)
                if count >= 0:
                    possible_counts.add(count)

# Answer queries
for _ in range(q):
    k = int(input())
    if k in possible_counts:
        print("Yes")
    else:
        print("No")