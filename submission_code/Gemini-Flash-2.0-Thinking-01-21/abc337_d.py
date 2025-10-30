import math

def solve():
    h, w, k = map(int, input().split())
    grid = [input() for _ in range(h)]
    min_operations = float('inf')
    found_solution = False
    
    for i in range(h):
        for j in range(w - k + 1):
            possible = True
            current_ops = 0
            for col in range(j, j + k):
                if grid[i][col] == 'x':
                    possible = False
                    break
                if grid[i][col] == '.':
                    current_ops += 1
            if possible:
                min_operations = min(min_operations, current_ops)
                found_solution = True
                
    for j in range(w):
        for i in range(h - k + 1):
            possible = True
            current_ops = 0
            for row in range(i, i + k):
                if grid[row][j] == 'x':
                    possible = False
                    break
                if grid[row][j] == '.':
                    current_ops += 1
            if possible:
                min_operations = min(min_operations, current_ops)
                found_solution = True
                
    if not found_solution:
        print(-1)
    else:
        print(min_operations)

if __name__ == '__main__':
    solve()