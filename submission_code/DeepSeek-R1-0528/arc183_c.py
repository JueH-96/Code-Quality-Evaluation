MOD = 998244353
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data: 
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    conditions_by_X = [[] for _ in range(n+1)]
    
    for _ in range(m):
        L = int(next(it))
        R = int(next(it))
        X = int(next(it))
        conditions_by_X[X].append((L, R))
    
    minR_arr = [[10**9] * (n+2) for _ in range(n+1)]
    
    for k in range(1, n+1):
        arr = minR_arr[k]
        for (L, R) in conditions_by_X[k]:
            if R < arr[L]:
                arr[L] = R
        for l in range(n, 0, -1):
            if arr[l] > arr[l+1]:
                arr[l] = arr[l+1]
                
    dp = [[0] * (n+2) for _ in range(n+2)]
    
    for i in range(n+2):
        for j in range(i):
            dp[i][j] = 1
                
    for length in range(1, n+1):
        for l in range(1, n - length + 2):
            r = l + length - 1
            total = 0
            for k in range(l, r+1):
                if minR_arr[k][l] <= r:
                    continue
                left_count = dp[l][k-1]
                right_count = dp[k+1][r]
                total = (total + left_count * right_count) % MOD
            dp[l][r] = total % MOD
            
    print(dp[1][n] % MOD)

if __name__ == '__main__':
    main()