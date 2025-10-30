def solve():
    n, w_limit = map(int, input().split())
    items = []
    for _ in range(n):
        weight, value = map(int, input().split())
        items.append({'weight': weight, 'value': value})
    
    dp = [[0] * (w_limit + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        item = items[i-1]
        weight_i = item['weight']
        value_i = item['value']
        for j in range(w_limit + 1):
            max_happiness = 0
            for k in range(j // weight_i + 1):
                current_happiness = (k * value_i - k**2) + dp[i-1][j - k * weight_i]
                max_happiness = max(max_happiness, current_happiness)
            dp[i][j] = max_happiness
            
    print(dp[n][w_limit])

if __name__ == '__main__':
    solve()