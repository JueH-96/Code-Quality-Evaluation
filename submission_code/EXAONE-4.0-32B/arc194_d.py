MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    s = data[1].strip()
    
    if n == 0:
        print(1)
        return
        
    ok = [[False] * n for _ in range(n)]
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n-1):
        j = i + 1
        if j < n and s[i] == '(' and s[j] == ')':
            ok[i][j] = True
            dp[i][j] = 1
            
    for length in range(4, n+1, 2):
        for i in range(0, n - length + 1):
            j = i + length - 1
            if s[i] == '(' and s[j] == ')' and ok[i+1][j-1]:
                ok[i][j] = True
            else:
                for k in range(i+1, j, 2):
                    if ok[i][k] and ok[k+1][j]:
                        ok[i][j] = True
                        break
    
    valid_ks = [[] for _ in range(n)]
    for i in range(n):
        for k in range(i+1, n, 2):
            if ok[i][k]:
                valid_ks[i].append(k)
                
    for length in range(4, n+1, 2):
        for i in range(0, n - length + 1):
            j = i + length - 1
            total = 0
            if s[i] == '(' and s[j] == ')':
                if i+1 <= j-1:
                    total = (total + dp[i+1][j-1]) % MOD
                else:
                    total = (total + 1) % MOD
                    
            for k in valid_ks[i]:
                if k >= j:
                    break
                left_val = 1
                if i+1 <= k-1:
                    left_val = dp[i+1][k-1]
                right_val = dp[k+1][j]
                total = (total + left_val * right_val) % MOD
                
            dp[i][j] = total % MOD
            
    print(dp[0][n-1])

if __name__ == "__main__":
    main()