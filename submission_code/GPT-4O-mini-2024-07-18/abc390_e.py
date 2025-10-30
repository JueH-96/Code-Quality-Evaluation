def max_min_vitamins(N, X, foods):
    # Separate foods by vitamin type
    vitamin_foods = {1: [], 2: [], 3: []}
    
    for v, a, c in foods:
        vitamin_foods[v].append((a, c))
    
    # Function to check if we can achieve at least `min_vitamin` of each vitamin
    def can_achieve(min_vitamin):
        dp = [float('-inf')] * (X + 1)
        dp[0] = 0  # 0 calories consumed gives 0 vitamins
        
        # Process each vitamin type
        for v in range(1, 4):
            current_dp = dp[:]
            for a, c in vitamin_foods[v]:
                for j in range(X - c + 1):
                    if current_dp[j] != float('-inf'):
                        current_dp[j + c] = max(current_dp[j + c], current_dp[j] + a)
            dp = current_dp
        
        # Check if we can achieve at least `min_vitamin` of each vitamin
        for j in range(X + 1):
            if dp[j] >= min_vitamin * 3:  # We need at least min_vitamin from each of the 3 vitamins
                return True
        return False
    
    # Binary search for the maximum minimum vitamin intake
    low, high = 0, 200000  # A_i can be at most 200000
    answer = 0
    
    while low <= high:
        mid = (low + high) // 2
        if can_achieve(mid):
            answer = mid  # We can achieve at least `mid`
            low = mid + 1
        else:
            high = mid - 1
    
    return answer

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N, X = map(int, data[0].split())
    foods = [tuple(map(int, line.split())) for line in data[1:N+1]]
    
    result = max_min_vitamins(N, X, foods)
    print(result)

if __name__ == "__main__":
    main()