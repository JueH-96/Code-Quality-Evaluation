# YOUR CODE HERE
def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))
    
    p = [x - 1 for x in p]
    q = [x - 1 for x in q]
    
    grid = [['0'] * n for _ in range(n)]
    
    for i in range(n):
        grid[p[i]][q[i]] = '1'
        
    for row in grid:
        print("".join(row))

solve()