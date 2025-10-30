n = int(input())
presses = []
for _ in range(n):
    line = input().split()
    a = int(line[0])
    s = line[1]
    presses.append((a, s))

# dp[left_pos][right_pos] = minimum fatigue with left hand at left_pos and right hand at right_pos
dp = [[0] * 101 for _ in range(101)]

for i in range(n):
    a, s = presses[i]
    new_dp = [[float('inf')] * 101 for _ in range(101)]
    
    if s == 'L':
        # Left hand presses key a
        for right_pos in range(1, 101):
            for prev_left_pos in range(1, 101):
                cost = abs(a - prev_left_pos)
                new_dp[a][right_pos] = min(new_dp[a][right_pos], dp[prev_left_pos][right_pos] + cost)
    else:
        # Right hand presses key a
        for left_pos in range(1, 101):
            for prev_right_pos in range(1, 101):
                cost = abs(a - prev_right_pos)
                new_dp[left_pos][a] = min(new_dp[left_pos][a], dp[left_pos][prev_right_pos] + cost)
    
    dp = new_dp

# Find the minimum fatigue at the end
result = float('inf')
for left_pos in range(1, 101):
    for right_pos in range(1, 101):
        result = min(result, dp[left_pos][right_pos])

print(result)