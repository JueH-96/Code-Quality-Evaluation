# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

dp = {}
dp[0] = 0

for i in range(n):
    new_dp = {}
    for prev_state, prev_exp in dp.items():
        # Option 1: Let the monster go
        new_dp[prev_state] = max(new_dp.get(prev_state, 0), prev_exp)
        
        # Option 2: Defeat the monster
        new_state = prev_state | (1 << i)
        defeated_count = bin(new_state).count('1')
        exp_gain = a[i]
        if defeated_count % 2 == 0:
            exp_gain *= 2
        new_dp[new_state] = max(new_dp.get(new_state, 0), prev_exp + exp_gain)
    dp = new_dp

print(max(dp.values()))