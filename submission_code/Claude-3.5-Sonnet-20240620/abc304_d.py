# YOUR CODE HERE
def solve():
    W, H = map(int, input().split())
    N = int(input())
    strawberries = [tuple(map(int, input().split())) for _ in range(N)]
    
    A = int(input())
    a = [0] + list(map(int, input().split())) + [W]
    
    B = int(input())
    b = [0] + list(map(int, input().split())) + [H]
    
    grid = [[0] * (A + 1) for _ in range(B + 1)]
    
    for x, y in strawberries:
        i = next(i for i in range(A + 1) if a[i] < x < a[i + 1])
        j = next(j for j in range(B + 1) if b[j] < y < b[j + 1])
        grid[j][i] += 1
    
    min_strawberries = min(min(row) for row in grid)
    max_strawberries = max(max(row) for row in grid)
    
    print(min_strawberries, max_strawberries)

solve()