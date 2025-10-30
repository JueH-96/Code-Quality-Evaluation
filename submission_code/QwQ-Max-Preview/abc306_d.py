n = int(input())
courses = [tuple(map(int, input().split())) for _ in range(n)]

dp0 = 0  # initial state is healthy with sum 0
dp1 = -float('inf')  # impossible to start with upset stomach

for x, y in courses:
    new_dp0 = -float('inf')
    new_dp1 = -float('inf')
    
    # Skip the current course
    new_dp0 = max(new_dp0, dp0)
    new_dp1 = max(new_dp1, dp1)
    
    # Eat the current course
    if x == 0:
        # Antidotal course can be eaten from any state, leading to healthy
        eat0 = max(dp0 + y, dp1 + y)
        new_dp0 = max(new_dp0, eat0)
    else:
        # Poisonous course can only be eaten from healthy state, leading to upset
        if dp0 != -float('inf'):
            eat1 = dp0 + y
            new_dp1 = max(new_dp1, eat1)
    
    # Update DP values for next iteration
    dp0, dp1 = new_dp0, new_dp1

# The answer is the maximum of the two possible states after all courses
ans = max(dp0, dp1)
print(ans if ans != -float('inf') else 0)