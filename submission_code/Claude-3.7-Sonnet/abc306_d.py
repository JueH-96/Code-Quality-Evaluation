def max_tastiness():
    n = int(input())
    courses = []
    for _ in range(n):
        x, y = map(int, input().split())
        courses.append((x, y))
    
    # Initialize DP arrays
    # dp_healthy[i] = max tastiness with healthy stomach after i courses
    # dp_upset[i] = max tastiness with upset stomach after i courses
    dp_healthy = [float('-inf')] * (n + 1)
    dp_upset = [float('-inf')] * (n + 1)
    dp_healthy[0] = 0  # Initially, Takahashi has a healthy stomach
    
    for i in range(1, n + 1):
        x, y = courses[i-1]
        
        # Skip the course
        dp_healthy[i] = dp_healthy[i-1]
        dp_upset[i] = dp_upset[i-1]
        
        # Eat the course
        if x == 0:  # Antidotal course
            # If stomach is healthy, it stays healthy
            dp_healthy[i] = max(dp_healthy[i], dp_healthy[i-1] + y)
            # If stomach is upset, it becomes healthy
            if dp_upset[i-1] != float('-inf'):
                dp_healthy[i] = max(dp_healthy[i], dp_upset[i-1] + y)
        else:  # Poisonous course
            # If stomach is healthy, it becomes upset
            dp_upset[i] = max(dp_upset[i], dp_healthy[i-1] + y)
            # If stomach is upset, he dies (not a valid state, so no update)
    
    # The answer is the maximum tastiness where he's still alive
    return max(0, max(dp_healthy[n], dp_upset[n]))

print(max_tastiness())