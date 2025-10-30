# YOUR CODE HERE
import sys

def main():
    N = int(sys.stdin.readline())
    grid = [sys.stdin.readline().strip() for _ in range(N)]
    
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
                # Calculate the number of valid triples where (i,j) is the common cell
                # For the row i, we have row_counts[i] - 1 choices for the second cell in the row
                # For the column j, we have col_counts[j] - 1 choices for the second cell in the column
                # The total is (row_counts[i] - 1) * (col_counts[j] - 1)
                total += (row_counts[i] - 1) * (col_counts[j] - 1)
    
    print(total)

if __name__ == "__main__":
    main()