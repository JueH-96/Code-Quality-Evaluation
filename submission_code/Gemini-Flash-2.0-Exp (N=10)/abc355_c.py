def solve():
    n, t = map(int, input().split())
    a = list(map(int, input().split()))
    
    grid = [[0] * n for _ in range(n)]
    marked = [[False] * n for _ in range(n)]
    
    for turn, num in enumerate(a):
        row = (num - 1) // n
        col = (num - 1) % n
        marked[row][col] = True
        
        # Check rows
        for r in range(n):
            if all(marked[r]):
                print(turn + 1)
                return
        
        # Check columns
        for c in range(n):
            if all(marked[r][c] for r in range(n)):
                print(turn + 1)
                return
        
        # Check diagonals
        if all(marked[i][i] for i in range(n)):
            print(turn + 1)
            return
        
        if all(marked[i][n - 1 - i] for i in range(n)):
            print(turn + 1)
            return
    
    print("-1")

solve()