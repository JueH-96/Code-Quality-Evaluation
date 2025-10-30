# YOUR CODE HERE
def solve():
    N, X, Y = map(int, input().split())
    dishes = []
    for _ in range(N):
        a, b = map(int, input().split())
        dishes.append((a, b))
    
    # dp[s][t] = maximum number of dishes we can eat with total sweetness s and saltiness t
    dp = {}
    dp[(0, 0)] = 0
    
    for i in range(N):
        a, b = dishes[i]
        new_dp = dp.copy()
        
        for (sweet, salt), count in dp.items():
            new_sweet = sweet + a
            new_salt = salt + b
            
            if new_sweet <= X and new_salt <= Y:
                key = (new_sweet, new_salt)
                if key not in new_dp or new_dp[key] < count + 1:
                    new_dp[key] = count + 1
        
        dp = new_dp
    
    print(max(dp.values()))

solve()