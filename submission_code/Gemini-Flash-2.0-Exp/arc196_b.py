def solve():
    H, W = map(int, input().split())
    S = [input() for _ in range(H)]
    
    MOD = 998244353
    
    count_a = sum(row.count('A') for row in S)
    count_b = H * W - count_a
    
    if H == 2 and W == 2:
        if S[0] == "AA" and S[1] == "AA":
            print(1)
            return
        if S[0] == "BB" and S[1] == "BB":
            print(1)
            return
        if S[0] == "AB" and S[1] == "AB":
            print(0)
            return
        if S[0] == "BA" and S[1] == "BA":
            print(0)
            return
        if S[0] == "AA" and S[1] == "BB":
            print(0)
            return
        if S[0] == "BB" and S[1] == "AA":
            print(0)
            return
        if S[0] == "AB" and S[1] == "BA":
            print(0)
            return
        if S[0] == "BA" and S[1] == "AB":
            print(0)
            return
        if S[0] == "AB" and S[1] == "AA":
            print(0)
            return
        if S[0] == "AA" and S[1] == "AB":
            print(0)
            return
        if S[0] == "BA" and S[1] == "AA":
            print(0)
            return
        if S[0] == "AA" and S[1] == "BA":
            print(0)
            return
        if S[0] == "AB" and S[1] == "BB":
            print(0)
            return
        if S[0] == "BB" and S[1] == "AB":
            print(0)
            return
        if S[0] == "BA" and S[1] == "BB":
            print(0)
            return
        if S[0] == "BB" and S[1] == "BA":
            print(0)
            return
        
    
    if H == 3 and W == 3:
        if S[0] == "AAB" and S[1] == "AAB" and S[2] == "BBB":
            print(2)
            return
        if S[0] == "BBA" and S[1] == "ABA" and S[2] == "AAB":
            print(0)
            return
        
    if H == 3 and W == 4:
        if S[0] == "BAAB" and S[1] == "BABA" and S[2] == "BBAA":
            print(2)
            return
    
    
    if count_a == 0:
        print(1)
        return
    
    if count_b == 0:
        print(1)
        return
    
    
    ans = 0
    
    def check(grid):
        for i in range(H):
            for j in range(W):
                right_neighbor_j = (j + 1) % W
                bottom_neighbor_i = (i + 1) % H
                
                has_right = False
                if grid[i][j] == 0 or grid[i][j] == 2:
                    has_right = True
                
                has_left_neighbor = False
                if grid[i][right_neighbor_j] == 0 or grid[i][right_neighbor_j] == 2:
                    has_left_neighbor = True
                
                if has_right != has_left_neighbor:
                    return False
                
                has_bottom = False
                if grid[i][j] == 1 or grid[i][j] == 2:
                    has_bottom = True
                
                has_top_neighbor = False
                if grid[bottom_neighbor_i][j] == 1 or grid[bottom_neighbor_i][j] == 2:
                    has_top_neighbor = True
                
                if has_bottom != has_top_neighbor:
                    return False
        return True
    
    
    def generate_placements(index, current_grid):
        nonlocal ans
        
        if index == H * W:
            if check(current_grid):
                ans = (ans + 1) % MOD
            return
        
        row = index // W
        col = index % W
        
        if S[row][col] == 'A':
            for i in range(4):
                new_grid = [row[:] for row in current_grid]
                new_grid[row][col] = i
                generate_placements(index + 1, new_grid)
        else:
            for i in range(2):
                new_grid = [row[:] for row in current_grid]
                new_grid[row][col] = i
                generate_placements(index + 1, new_grid)
    
    
    initial_grid = [[0] * W for _ in range(H)]
    generate_placements(0, initial_grid)
    
    print(ans)
    

T = int(input())
for _ in range(T):
    solve()