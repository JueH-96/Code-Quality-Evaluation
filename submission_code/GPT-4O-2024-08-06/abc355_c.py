# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:2+T]))
    
    # Initialize the grid marking
    marked = [[False] * N for _ in range(N)]
    
    # Function to convert number to grid coordinates
    def number_to_coordinates(num):
        num -= 1
        return num // N, num % N
    
    # Check if any row, column, or diagonal is fully marked
    def check_bingo():
        # Check rows and columns
        for i in range(N):
            if all(marked[i][j] for j in range(N)):
                return True
            if all(marked[j][i] for j in range(N)):
                return True
        
        # Check diagonals
        if all(marked[i][i] for i in range(N)):
            return True
        if all(marked[i][N-1-i] for i in range(N)):
            return True
        
        return False
    
    # Process each turn
    for turn in range(T):
        num = A[turn]
        x, y = number_to_coordinates(num)
        marked[x][y] = True
        
        if check_bingo():
            print(turn + 1)
            return
    
    # If no bingo is achieved
    print(-1)

if __name__ == "__main__":
    main()