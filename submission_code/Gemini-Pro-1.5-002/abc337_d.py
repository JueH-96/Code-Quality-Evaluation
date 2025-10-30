# YOUR CODE HERE
def solve():
    h, w, k = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    ans = float('inf')
    
    for i in range(1 << (h * w)):
        cost = 0
        temp_grid = [list(row) for row in grid]
        
        for r in range(h):
            for c in range(w):
                if (i >> (r * w + c)) & 1:
                    if temp_grid[r][c] == '.':
                        temp_grid[r][c] = 'o'
                        cost += 1

        found = False
        for r in range(h):
            for c in range(w - k + 1):
                all_o = True
                for x in range(k):
                    if temp_grid[r][c + x] != 'o':
                        all_o = False
                        break
                if all_o:
                    found = True
                    break
            if found:
                break
        
        if found:
            ans = min(ans, cost)
            continue
            
        for r in range(h - k + 1):
            for c in range(w):
                all_o = True
                for x in range(k):
                    if temp_grid[r + x][c] != 'o':
                        all_o = False
                        break
                if all_o:
                    found = True
                    break
            if found:
                break

        if found:
            ans = min(ans, cost)

    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

solve()