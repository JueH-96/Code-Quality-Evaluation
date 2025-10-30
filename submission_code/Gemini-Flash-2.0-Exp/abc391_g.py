def solve():
    n, m = map(int, input().split())
    s = input()
    
    mod = 998244353
    
    def calculate_lcs(str1, str2):
        len1 = len(str1)
        len2 = len(str2)
        
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[len1][len2]
    
    ans = []
    for k in range(n + 1):
        count = 0
        for i in range(26**m):
            temp = i
            t = ""
            for _ in range(m):
                t = chr(ord('a') + (temp % 26)) + t
                temp //= 26
            
            lcs_len = calculate_lcs(s, t)
            if lcs_len == k:
                count = (count + 1) % mod
        ans.append(count)
    
    print(*ans)

solve()