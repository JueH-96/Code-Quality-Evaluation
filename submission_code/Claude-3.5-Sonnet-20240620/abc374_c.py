# YOUR CODE HERE
def min_max_lunch_break(N, K):
    total_people = sum(K)
    target = total_people // 2

    def dfs(index, current_sum):
        if index == N:
            return current_sum
        
        # Try including the current department
        include = dfs(index + 1, current_sum + K[index])
        
        # Try excluding the current department
        exclude = dfs(index + 1, current_sum)
        
        # Return the sum closest to the target
        return max(include, exclude) if abs(include - target) < abs(exclude - target) else min(include, exclude)

    best_sum = dfs(0, 0)
    return max(best_sum, total_people - best_sum)

# Read input
N = int(input())
K = list(map(int, input().split()))

# Solve and print output
result = min_max_lunch_break(N, K)
print(result)