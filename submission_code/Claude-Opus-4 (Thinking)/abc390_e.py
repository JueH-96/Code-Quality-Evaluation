# YOUR CODE HERE
N, X = map(int, input().split())
foods = []
for _ in range(N):
    v, a, c = map(int, input().split())
    foods.append((v-1, a, c))  # Convert to 0-indexed vitamin type

# dp[c] = set of achievable (v1, v2, v3) tuples with exactly c calories
dp = {0: {(0, 0, 0)}}

for v_type, amount, calories in foods:
    new_dp = {}
    
    for c, tuples in dp.items():
        # Don't include this food
        if c not in new_dp:
            new_dp[c] = set()
        new_dp[c].update(tuples)
        
        # Include this food
        if c + calories <= X:
            if c + calories not in new_dp:
                new_dp[c + calories] = set()
            for v1, v2, v3 in tuples:
                new_v = [v1, v2, v3]
                new_v[v_type] += amount
                new_dp[c + calories].add(tuple(new_v))
    
    dp = new_dp

# Find the maximum minimum among all achievable combinations
ans = 0
for c, tuples in dp.items():
    for v1, v2, v3 in tuples:
        ans = max(ans, min(v1, v2, v3))

print(ans)