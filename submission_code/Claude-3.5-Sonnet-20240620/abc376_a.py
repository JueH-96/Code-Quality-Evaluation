# YOUR CODE HERE
def count_candies(N, C, T):
    candies = 0
    last_candy_time = -float('inf')
    
    for time in T:
        if time - last_candy_time >= C:
            candies += 1
            last_candy_time = time
    
    return candies

# Read input
N, C = map(int, input().split())
T = list(map(int, input().split()))

# Calculate and print the result
result = count_candies(N, C, T)
print(result)