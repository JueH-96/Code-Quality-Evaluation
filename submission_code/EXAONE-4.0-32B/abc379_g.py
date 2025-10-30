mod = 998244353

import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    if H < W:
        new_grid = []
        for j in range(W):
            s = ''.join(grid[i][j] for i in range(H))
            new_grid.append(s)
        grid = new_grid
        H, W = W, H
        
    nrows = H
    ncols = W
    
    power3 = [1] * (ncols+1)
    for i in range(1, ncols+1):
        power3[i] = power3[i-1] * 3
        
    valid = []
    for i in range(nrows):
        arr = [True] * power3[ncols]
        for mask in range(power3[ncols]):
            colors = []
            temp = mask
            for j in range(ncols):
                colors.append(temp % 3)
                temp //= 3
            for j in range(ncols):
                c_char = chr(ord('1') + colors[j])
                if grid[i][j] != '?' and grid[i][j] != c_char:
                    arr[mask] = False
                    break
            if not arr[mask]:
                continue
            for j in range(1, ncols):
                if colors[j] == colors[j-1]:
                    arr[mask] = False
                    break
        valid.append(arr)
    
    dp = [0] * power3[ncols]
    for mask in range(power3[ncols]):
        if valid[0][mask]:
            dp[mask] = 1
            
    if nrows == 1:
        print(sum(dp) % mod)
        return
        
    for i in range(1, nrows):
        temp_arr = dp[:]
        for j in range(ncols):
            new_temp = [0] * power3[ncols]
            for base in range(power3[ncols] // 3):
                high_part = base // power3[j]
                low_part = base % power3[j]
                mask0 = high_part * power3[j+1] + low_part
                mask1 = mask0 + power3[j]
                mask2 = mask1 + power3[j]
                x0 = temp_arr[mask0]
                x1 = temp_arr[mask1]
                x2 = temp_arr[mask2]
                y0 = (x1 + x2) % mod
                y1 = (x0 + x2) % mod
                y2 = (x0 + x1) % mod
                new_temp[mask0] = y0
                new_temp[mask1] = y1
                new_temp[mask2] = y2
            temp_arr = new_temp
            
        for mask in range(power3[ncols]):
            if valid[i][mask]:
                dp[mask] = temp_arr[mask] % mod
            else:
                dp[mask] = 0
                
    ans = sum(dp) % mod
    print(ans)

if __name__ == '__main__':
    main()