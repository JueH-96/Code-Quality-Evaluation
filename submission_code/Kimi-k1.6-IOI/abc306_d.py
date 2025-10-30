n = int(input())
courses = [tuple(map(int, input().split())) for _ in range(n)]

prev_healthy = 0
prev_upset = -10**18

for x, y in courses:
    new_healthy = prev_healthy  # Skip the current course
    if x == 0:
        # Eating the course from healthy or upset state
        candidate1 = prev_healthy + y
        candidate2 = prev_upset + y
        new_healthy = max(new_healthy, candidate1, candidate2)
    
    new_upset = prev_upset  # Skip the current course
    if x == 1:
        # Eating the course from healthy state
        candidate = prev_healthy + y
        new_upset = max(new_upset, candidate)
    
    # Update the states for the next iteration
    prev_healthy, prev_upset = new_healthy, new_upset

# The answer is the maximum of the two states and zero
print(max(prev_healthy, prev_upset, 0))