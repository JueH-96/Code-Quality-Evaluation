# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    grid = data[1:N+1]
    
    # Precompute the number of 'o's in each row and column
    row_counts = [0] * N
    col_counts = [0] * N
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                row_counts[i] += 1
                col_counts[j] += 1
    
    total = 0
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                # Calculate the number of valid triples where (i,j) is the intersection
                # We need to choose one 'o' from the same row and one 'o' from the same column
                # The number of 'o's in row i is row_counts[i], but we exclude (i,j)
                # Similarly for column j
                row_o = row_counts[i] - 1
                col_o = col_counts[j] - 1
                total += row_o * col_o
    
    print(total)

if __name__ == "__main__":
    main()