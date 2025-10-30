import sys

mod = 998244353

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    exist = [[] for _ in range(n+1)]
    
    for _ in range(m):
        L = int(next(it))
        R = int(next(it))
        X = int(next(it))
        exist[X].append((L, R))
        
    maxL_list = [None] * (n+1)
    
    for p in range(1, n+1):
        conds = exist[p]
        if not conds:
            maxL_arr = [0] * (n+1)
            maxL_list[p] = maxL_arr
            continue
            
        conds_sorted = sorted(conds, key=lambda x: x[1])
        maxL_arr = [0] * (n+1)
        cur_max = 0
        idx = 0
        for r_val in range(1, n+1):
            while idx < len(conds_sorted) and conds_sorted[idx][1] == r_val:
                L0 = conds_sorted[idx][0]
                if L0 > cur_max:
                    cur_max = L0
                idx += 1
            maxL_arr[r_val] = cur_max
        maxL_list[p] = maxL_arr
        
    dp = [[0] * (n+1) for _ in range(n+1)]
    for i in range(1, n+2):
        if i-1 >= 0 and i-1 <= n:
            dp[i][i-1] = 1
            
    for length in range(1, n+1):
        for l in range(1, n - length + 2):
            r = l + length - 1
            total = 0
            for p in range(l, r+1):
                if maxL_list[p][r] >= l:
                    continue
                left_val = dp[l][p-1]
                right_val = dp[p+1][r] if p+1 <= n else 1
                total = (total + left_val * right_val) % mod
            dp[l][r] = total
            
    print(dp[1][n] % mod)

if __name__ == '__main__':
    main()