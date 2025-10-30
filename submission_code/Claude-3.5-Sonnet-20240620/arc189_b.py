# YOUR CODE HERE
def min_coordinate_sum(n, coordinates):
    coordinates.sort()
    dp = [0] * n
    
    for i in range(n):
        dp[i] = coordinates[i]
        if i >= 3:
            dp[i] = min(dp[i], dp[i-3] + (coordinates[i] - coordinates[i-3]) // 2)
    
    return sum(dp)

# Read input
n = int(input())
coordinates = list(map(int, input().split()))

# Calculate and print the result
result = min_coordinate_sum(n, coordinates)
print(result)