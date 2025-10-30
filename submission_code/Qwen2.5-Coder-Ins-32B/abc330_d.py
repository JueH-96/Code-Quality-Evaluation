# YOUR CODE HERE
import sys

def count_valid_triples(N, grid):
    row_counts = [{} for _ in range(N)]
    col_counts = [{} for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                row_counts[i][j] = row_counts[i].get(j, 0) + 1
                col_counts[j][i] = col_counts[j].get(i, 0) + 1
    
    count = 0
    
    for i in range(N):
        for j1 in row_counts[i]:
            for j2 in row_counts[i]:
                if j1 < j2:
                    for i1 in col_counts[j1]:
                        for i2 in col_counts[j2]:
                            if i1 < i2 and i1 != i and i2 != i and grid[i1][j2] == 'o' and grid[i2][j1] == 'o':
                                count += 1
    
    return count

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    grid = input[1:]
    result = count_valid_triples(N, grid)
    print(result)

if __name__ == "__main__":
    main()