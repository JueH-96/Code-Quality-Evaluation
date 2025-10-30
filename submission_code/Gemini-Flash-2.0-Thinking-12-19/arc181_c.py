def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))
    
    row_rank = [0] * n
    col_rank = [0] * n
    
    for i in range(n):
        row_rank[p[i]-1] = i + 1
    for i in range(n):
        col_rank[q[i]-1] = i + 1
        
    grid = [['' for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if row_rank[i] + col_rank[j] <= n:
                grid[i][j] = '0'
            else:
                grid[i][j] = '1'
                
    for i in range(n):
        print("".join(grid[i]))

if __name__ == '__main__':
    solve()