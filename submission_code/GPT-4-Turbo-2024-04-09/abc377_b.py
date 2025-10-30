# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # Initialize the board
    board = data
    
    # Determine the rows and columns that are threatened by existing pieces
    threatened_rows = set()
    threatened_cols = set()
    
    for i in range(8):
        for j in range(8):
            if board[i][j] == '#':
                threatened_rows.add(i)
                threatened_cols.add(j)
    
    # Count the number of safe empty squares
    safe_count = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] == '.' and i not in threatened_rows and j not in threatened_cols:
                safe_count += 1
    
    # Output the result
    print(safe_count)

if __name__ == "__main__":
    main()