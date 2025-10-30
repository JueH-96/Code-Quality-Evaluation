def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    def is_valid(r, c):
        return 0 <= r < H and 0 <= c < W
    
    def find_path(r, c, path_len):
        if r == H - 1 and c == W - 1:
            return True
        
        snuke = "snuke"
        expected_char = snuke[path_len % 5]
        
        if grid[r][c] != expected_char:
            return False
        
        # Explore adjacent cells
        neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        
        for nr, nc in neighbors:
            if is_valid(nr, nc):
                
                if find_path(nr, nc, path_len + 1):
                    return True
        
        return False
    
    if find_path(0, 0, 0):
        print("Yes")
    else:
        print("No")

solve()