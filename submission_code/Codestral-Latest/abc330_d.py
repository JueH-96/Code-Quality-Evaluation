# YOUR CODE HERE
import sys
from collections import defaultdict

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
grid = data[1:N+1]

# Count the number of 'o' in each row and column
row_count = [0] * N
col_count = [0] * N

for i in range(N):
    for j in range(N):
        if grid[i][j] == 'o':
            row_count[i] += 1
            col_count[j] += 1

# Count the number of valid triples
count = 0

for i in range(N):
    for j in range(N):
        if grid[i][j] == 'o':
            # Count the number of 'o' in the same row and different column
            same_row_diff_col = row_count[i] - 1
            # Count the number of 'o' in the same column and different row
            same_col_diff_row = col_count[j] - 1
            # Add the product of these counts to the total count
            count += same_row_diff_col * same_col_diff_row

print(count)