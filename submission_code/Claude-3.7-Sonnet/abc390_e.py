def max_min_vitamin_intake(N, X, foods):
    # Group foods by vitamin type
    foods_by_vitamin = {1: [], 2: [], 3: []}
    for v, a, c in foods:
        foods_by_vitamin[v].append((a, c))
    
    # If any vitamin is missing, return 0
    for v in range(1, 4):
        if v not in foods_by_vitamin or not foods_by_vitamin[v]:
            return 0
    
    # Compute minimum calories needed to get at least j units of each vitamin
    min_calorie_by_vitamin = {}
    max_amounts = {}
    
    for v in range(1, 4):
        foods_v = foods_by_vitamin[v]
        
        # dp[j] = maximum amount of vitamin we can get using at most j calories
        dp = [0] * (X + 1)
        
        for a, c in foods_v:
            for j in range(X, c - 1, -1):
                dp[j] = max(dp[j], dp[j-c] + a)
        
        # Compute minimum calorie needed to get at least j units of the vitamin
        max_amount = dp[X]
        max_amounts[v] = max_amount
        min_calorie = [float('inf')] * (max_amount + 1)
        min_calorie[0] = 0
        
        for j in range(1, max_amount + 1):
            # Binary search for minimum calorie c such that dp[c] >= j
            left, right = 0, X
            while left < right:
                mid = (left + right) // 2
                if dp[mid] >= j:
                    right = mid
                else:
                    left = mid + 1
            min_calorie[j] = left
        
        min_calorie_by_vitamin[v] = min_calorie
    
    # Binary search for maximum value of minimum vitamin intake
    max_possible_min = min(max_amounts.values())
    
    left, right = 0, max_possible_min
    result = 0
    
    while left <= right:
        mid = (left + right) // 2
        
        total_calorie = sum(min_calorie_by_vitamin[v][mid] for v in range(1, 4))
        
        if total_calorie <= X:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def main():
    N, X = map(int, input().split())
    foods = []
    for _ in range(N):
        V, A, C = map(int, input().split())
        foods.append((V, A, C))
    
    result = max_min_vitamin_intake(N, X, foods)
    print(result)

if __name__ == "__main__":
    main()