def find_components(grid, H, W):
    def dfs(i, j, visited):
        if i < 0 or i >= H or j < 0 or j >= W or grid[i][j] == '.' or (i,j) in visited:
            return
        visited.add((i,j))
        for ni, nj in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
            dfs(ni, nj, visited)
            
    visited = set()
    components = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and (i,j) not in visited:
                dfs(i, j, visited)
                components += 1
    return components

def count_components_if_painted(grid, H, W, x, y):
    grid[x][y] = '#'
    count = find_components(grid, H, W)
    grid[x][y] = '.'
    return count

def mod_inverse(a, m):
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a, m)
    return (x % m + m) % m

H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]

MOD = 998244353
total_red = sum(1 for i in range(H) for j in range(W) if grid[i][j] == '.')
numerator = 0

for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            numerator += count_components_if_painted(grid, H, W)

denominator = total_red
inv_denominator = mod_inverse(denominator, MOD)
result = (numerator * inv_denominator) % MOD

print(result)