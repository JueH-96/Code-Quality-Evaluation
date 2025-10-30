def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))
    
    p_inv = [0] * (n + 1)
    for i in range(n):
        p_inv[p[i]] = i + 1
        
    q_inv = [0] * (n + 1)
    for i in range(n):
        q_inv[q[i]] = i + 1
        
    grid = [['' for _ in range(n)] for _ in range(n)]
    
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            i = p_inv[r]
            j = q_inv[c]
            if i + j <= n:
                grid[r-1][c-1] = '0'
            else:
                grid[r-1][c-1] = '1'
                
    for r in range(n):
        print("".join(grid[r]))

if __name__ == '__main__':
    solve()