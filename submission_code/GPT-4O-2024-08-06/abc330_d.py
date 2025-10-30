# YOUR CODE HERE
def count_triples(N, grid):
    # Precompute the positions of 'o' in each row and column
    row_o_positions = [[] for _ in range(N)]
    col_o_positions = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                row_o_positions[i].append(j)
                col_o_positions[j].append(i)
    
    count = 0
    
    # Iterate over each row to find pairs of 'o's
    for i in range(N):
        row_o = row_o_positions[i]
        # For each pair of 'o's in the same row
        for k in range(len(row_o)):
            for l in range(k + 1, len(row_o)):
                j1 = row_o[k]
                j2 = row_o[l]
                
                # Find third 'o' in the same column as j1 or j2
                # Exclude the current row i
                count += len(col_o_positions[j1]) - 1
                count += len(col_o_positions[j2]) - 1
    
    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
grid = data[1:N+1]

result = count_triples(N, grid)
print(result)