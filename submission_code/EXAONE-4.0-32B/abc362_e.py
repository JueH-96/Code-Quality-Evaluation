mod = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    ans = [0] * (n + 1)
    ans[1] = n
    
    if n > 1:
        dp = [dict() for _ in range(n)]
        
        for i in range(1, n):
            for j in range(0, i):
                d = A[i] - A[j]
                
                if d not in dp[i]:
                    dp[i][d] = [0] * (n + 1)
                    
                dp[i][d][2] = (dp[i][d][2] + 1) % mod
                
                if d in dp[j]:
                    arr_j = dp[j][d]
                    for l in range(2, n):
                        if arr_j[l] != 0:
                            new_len = l + 1
                            dp[i][d][new_len] = (dp[i][d][new_len] + arr_j[l]) % mod
        
        for i in range(n):
            for d, arr in dp[i].items():
                for l in range(2, n + 1):
                    ans[l] = (ans[l] + arr[l]) % mod
    
    res = [str(ans[k] % mod) for k in range(1, n + 1)]
    print(" ".join(res))

if __name__ == '__main__':
    main()