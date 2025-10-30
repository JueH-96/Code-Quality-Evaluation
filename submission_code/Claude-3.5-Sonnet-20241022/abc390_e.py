def solve():
    # Read input
    N, X = map(int, input().split())
    foods = []
    for _ in range(N):
        v, a, c = map(int, input().split())
        foods.append((v-1, a, c))  # Convert vitamin numbers to 0-based indexing
    
    # Binary search on the minimum vitamin amount
    def check(target):
        # For each possible combination of foods that gives at least 'target' of each vitamin
        # Try to find one that uses at most X calories
        
        # Group foods by vitamin type
        vitamin_groups = [[] for _ in range(3)]
        for v, a, c in foods:
            if a >= target:  # Only consider foods that give enough of their vitamin
                vitamin_groups[v].append((a, c))
        
        # Try all combinations of foods for each vitamin
        min_calories = float('inf')
        
        # For each food choice for vitamin 1
        for i in range(len(vitamin_groups[0]) + 1):
            if i == 0:
                v1_amount = 0
                v1_calories = 0
            else:
                v1_amount = vitamin_groups[0][i-1][0]
                v1_calories = vitamin_groups[0][i-1][1]
            
            # For each food choice for vitamin 2
            for j in range(len(vitamin_groups[1]) + 1):
                if j == 0:
                    v2_amount = 0
                    v2_calories = 0
                else:
                    v2_amount = vitamin_groups[1][j-1][0]
                    v2_calories = vitamin_groups[1][j-1][1]
                
                # For each food choice for vitamin 3
                for k in range(len(vitamin_groups[2]) + 1):
                    if k == 0:
                        v3_amount = 0
                        v3_calories = 0
                    else:
                        v3_amount = vitamin_groups[2][k-1][0]
                        v3_calories = vitamin_groups[2][k-1][1]
                    
                    total_calories = v1_calories + v2_calories + v3_calories
                    
                    # If this combination is valid
                    if (v1_amount >= target or i == 0) and \
                       (v2_amount >= target or j == 0) and \
                       (v3_amount >= target or k == 0) and \
                       not (i == 0 and j == 0 and k == 0):
                        min_calories = min(min_calories, total_calories)
        
        return min_calories <= X
    
    # Binary search
    left = 0
    right = 2 * 10**5
    
    while right - left > 1:
        mid = (left + right) // 2
        if check(mid):
            left = mid
        else:
            right = mid
    
    print(left)

solve()