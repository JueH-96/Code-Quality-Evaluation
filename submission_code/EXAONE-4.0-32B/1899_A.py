t = int(input().strip())
results = []
for _ in range(t):
    n = int(input().strip())
    min_x = n - 11
    max_x = n + 11
    size = max_x - min_x + 1
    dp = [[''] * size for _ in range(11)]
    
    for x in range(min_x, max_x + 1):
        idx = x - min_x
        dp[10][idx] = 'Second'
        
    for k in range(9, -1, -1):
        for x in range(min_x, max_x + 1):
            idx = x - min_x
            if k % 2 == 0:
                if (x + 1) % 3 == 0 or (x - 1) % 3 == 0:
                    dp[k][idx] = 'First'
                else:
                    idx1 = (x + 1) - min_x
                    idx2 = (x - 1) - min_x
                    if dp[k+1][idx1] == 'First' or dp[k+1][idx2] == 'First':
                        dp[k][idx] = 'First'
                    else:
                        dp[k][idx] = 'Second'
            else:
                idx1 = (x + 1) - min_x
                idx2 = (x - 1) - min_x
                if dp[k+1][idx1] == 'Second' or dp[k+1][idx2] == 'Second':
                    dp[k][idx] = 'Second'
                else:
                    dp[k][idx] = 'First'
                    
    initial_idx = n - min_x
    results.append(dp[0][initial_idx])

for res in results:
    print(res)