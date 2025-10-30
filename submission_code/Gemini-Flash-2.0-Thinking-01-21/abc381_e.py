def solve():
    n, q = map(int, input().split())
    s = input()
    queries = []
    for _ in range(q):
        l, r = map(int, input().split())
        queries.append({'l': l, 'r': r})
    
    results = []
    for query in queries:
        l, r = query['l'], query['r']
        substring = s[l-1:r]
        substring_len = len(substring)
        if substring_len == 0:
            results.append(0)
            continue
            
        dp = [[-1] * 3 for _ in range(substring_len + 1)]
        dp[0][0] = 0
        dp[0][1] = -1
        dp[0][2] = -1
        
        for i in range(1, substring_len + 1):
            char = substring[i-1]
            dp[i][0] = dp[i-1][0]
            dp[i][1] = dp[i-1][1]
            dp[i][2] = dp[i-1][2]
            
            if char == '1':
                if dp[i-1][0] != -1:
                    dp[i][0] = max(dp[i][0], dp[i-1][0] + 1)
            elif char == '/':
                if dp[i-1][0] != -1:
                    dp[i][1] = max(dp[i][1], dp[i-1][0] + 1)
            elif char == '2':
                if dp[i-1][2] != -1:
                    dp[i][2] = max(dp[i][2], dp[i-1][2] + 1)
                if dp[i-1][1] != -1:
                    dp[i][2] = max(dp[i][2], dp[i-1][1] + 1)
                    
        max_len = 0
        if dp[substring_len][1] != -1:
            max_len = max(max_len, dp[substring_len][1])
        if dp[substring_len][2] != -1:
            max_len = max(max_len, dp[substring_len][2])
            
        results.append(max_len)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()