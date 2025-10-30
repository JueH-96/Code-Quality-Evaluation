import sys

def count_valid_triples(N, grid):
    row_counts = [0] * N
    col_counts = [0] * N
    
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                row_counts[i] += 1
                col_counts[j] += 1
    
    total_triples = 0
    
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                # Calculate the number of valid triples with (i, j) as the third cell
                # that is not in the same row or column as the other two cells
                row_others = row_counts[i] - 1
                col_others = col_counts[j] - 1
                total_triples += row_others * col_others
    
    return total_triples

if __name__ == "__main__":
    N = int(input())
    grid = [input().strip() for _ in range(N)]
    result = count_valid_triples(N, grid)
    print(result)