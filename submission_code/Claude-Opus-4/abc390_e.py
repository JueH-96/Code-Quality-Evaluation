def can_achieve_min_vitamin(foods, X, target):
    # Group foods by vitamin type
    vitamin_foods = [[], [], []]
    for v, a, c in foods:
        vitamin_foods[v-1].append((a, c))
    
    # For each vitamin type, find minimum calories needed to reach target
    min_calories = [float('inf')] * 3
    
    for vitamin_type in range(3):
        if not vitamin_foods[vitamin_type]:
            # If no food provides this vitamin, we can't achieve any positive target
            if target > 0:
                return False
            min_calories[vitamin_type] = 0
            continue
        
        # Dynamic programming to find minimum calories for this vitamin
        foods_for_vitamin = vitamin_foods[vitamin_type]
        n = len(foods_for_vitamin)
        
        # dp[i] = minimum calories to get at least i units of this vitamin
        max_vitamin = sum(a for a, c in foods_for_vitamin)
        if max_vitamin < target:
            return False
        
        dp = [float('inf')] * (target + 1)
        dp[0] = 0
        
        for amount, calories in foods_for_vitamin:
            # Process in reverse to avoid using same food multiple times
            for i in range(min(target, max_vitamin), -1, -1):
                if dp[i] < float('inf') and i + amount <= target:
                    dp[min(i + amount, target)] = min(dp[min(i + amount, target)], dp[i] + calories)
        
        min_calories[vitamin_type] = dp[target]
    
    # Check if total minimum calories is within budget
    return sum(min_calories) <= X

# Read input
N, X = map(int, input().split())
foods = []
for _ in range(N):
    v, a, c = map(int, input().split())
    foods.append((v, a, c))

# Binary search on the answer
left, right = 0, 200000 * N
result = 0

while left <= right:
    mid = (left + right) // 2
    if can_achieve_min_vitamin(foods, X, mid):
        result = mid
        left = mid + 1
    else:
        right = mid - 1

print(result)