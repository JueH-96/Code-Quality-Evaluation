# YOUR CODE HERE
def solve():
    n, t = map(int, input().split())
    a = list(map(int, input().split()))

    grid = [[0] * n for _ in range(n)]
    
    for turn, num in enumerate(a, 1):
        r = (num - 1) // n
        c = (num - 1) % n
        grid[r][c] = 1

        # Check rows
        for i in range(n):
            if all(grid[i]):
                print(turn)
                return

        # Check columns
        for j in range(n):
            if all(grid[i][j] for i in range(n)):
                print(turn)
                return

        # Check diagonals
        if all(grid[i][i] for i in range(n)):
            print(turn)
            return
        
        if all(grid[i][n-1-i] for i in range(n)):
            print(turn)
            return
            
    print(-1)

solve()