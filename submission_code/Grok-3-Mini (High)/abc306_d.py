import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
NEG_INF = -400000000000000  # -4e14, a value smaller than any possible sum
prev_healthy = 0  # Starting state: healthy with tastiness 0
prev_upset = NEG_INF  # Starting state: upset is impossible

for _ in range(N):
    X = int(data[index])
    index += 1
    Y = int(data[index])
    index += 1
    
    # Compute new healthy state
    new_healthy = NEG_INF
    if prev_healthy != NEG_INF:
        new_healthy = max(new_healthy, prev_healthy)  # Skip and was healthy
    if X == 0:  # Antidotal course
        if prev_healthy != NEG_INF:
            new_healthy = max(new_healthy, prev_healthy + Y)  # Eat from healthy
        if prev_upset != NEG_INF:
            new_healthy = max(new_healthy, prev_upset + Y)  # Eat from upset
    
    # Compute new upset state
    new_upset = NEG_INF
    if prev_upset != NEG_INF:
        new_upset = max(new_upset, prev_upset)  # Skip and was upset
    if X == 1:  # Poisonous course
        if prev_healthy != NEG_INF:
            new_upset = max(new_upset, prev_healthy + Y)  # Eat from healthy
    
    # Update previous states to new states
    prev_healthy = new_healthy
    prev_upset = new_upset

# The answer is the maximum of the two states after all courses
answer = max(prev_healthy, prev_upset)
print(answer)