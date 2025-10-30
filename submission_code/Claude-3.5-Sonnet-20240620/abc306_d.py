# YOUR CODE HERE
N = int(input())

# Initialize dp arrays for healthy and upset stomach states
dp_healthy = [0] * (N + 1)
dp_upset = [0] * (N + 1)

for i in range(1, N + 1):
    X, Y = map(int, input().split())
    
    if X == 0:  # Antidotal course
        # If we eat it with a healthy stomach, we stay healthy
        dp_healthy[i] = max(dp_healthy[i-1] + Y, dp_healthy[i-1])
        
        # If we eat it with an upset stomach, we become healthy
        dp_healthy[i] = max(dp_healthy[i], dp_upset[i-1] + Y)
        
        # If we skip it with an upset stomach, we stay upset
        dp_upset[i] = dp_upset[i-1]
    else:  # Poisonous course
        # If we eat it with a healthy stomach, we become upset
        dp_upset[i] = max(dp_healthy[i-1] + Y, dp_upset[i-1])
        
        # If we skip it with a healthy stomach, we stay healthy
        dp_healthy[i] = dp_healthy[i-1]

# The answer is the maximum tastiness when ending with a healthy stomach
print(max(dp_healthy[N], dp_upset[N]))