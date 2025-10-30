# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    T = int(input[1])
    A = list(map(int, input[2:]))

    # Initialize the grid and marks
    grid = [[N * i + j + 1 for j in range(N)] for i in range(N)]
    marked_rows = [0] * N
    marked_cols = [0] * N
    marked_diag1 = 0
    marked_diag2 = 0

    # Create a reverse mapping from number to (row, col)
    num_to_pos = {}
    for i in range(N):
        for j in range(N):
            num_to_pos[grid[i][j]] = (i, j)

    # Process each turn
    for turn in range(T):
        num = A[turn]
        i, j = num_to_pos[num]
        marked_rows[i] += 1
        marked_cols[j] += 1
        if i == j:
            marked_diag1 += 1
        if i + j == N - 1:
            marked_diag2 += 1

        # Check for Bingo
        if marked_rows[i] == N or marked_cols[j] == N or marked_diag1 == N or marked_diag2 == N:
            print(turn + 1)
            return

    print(-1)

if __name__ == "__main__":
    main()