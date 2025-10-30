import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print("No")
        return
    
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    top_black = H
    bottom_black = -1
    left_black = W
    right_black = -1
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < top_black:
                    top_black = i
                if i > bottom_black:
                    bottom_black = i
                if j < left_black:
                    left_black = j
                if j > right_black:
                    right_black = j
                    
    if top_black == H:
        print("No")
        return
        
    L_arr = [0] * H
    R_arr = [0] * H
    
    for i in range(H):
        has_white = False
        for j in range(left_black, right_black + 1):
            if grid[i][j] == '.':
                has_white = True
                break
                
        if has_white:
            L_arr[i] = 1
            R_arr[i] = 0
        else:
            j = left_black
            while j >= 0 and grid[i][j] != '.':
                j -= 1
            L_arr[i] = j + 1
            
            j = right_black
            while j < W and grid[i][j] != '.':
                j += 1
            R_arr[i] = j - 1
            
    n = H
    if n == 0:
        print("No")
        return
        
    dp_max = [[0] * n for _ in range(n)]
    dp_min = [[0] * n for _ in range(n)]
    
    for i in range(n):
        dp_max[i][i] = L_arr[i]
        dp_min[i][i] = R_arr[i]
        
    for i in range(n):
        for j in range(i+1, n):
            dp_max[i][j] = max(dp_max[i][j-1], L_arr[j])
            dp_min[i][j] = min(dp_min[i][j-1], R_arr[j])
            
    found = False
    for a in range(0, top_black + 1):
        for b in range(bottom_black, n):
            global_L = dp_max[a][b]
            global_R = dp_min[a][b]
            if global_L <= left_black and global_R >= right_black and global_L <= global_R:
                found = True
                break
        if found:
            break
            
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()