def max_tastiness(N, courses):
    healthy = 0  # Maximum tastiness when healthy
    upset = float('-inf')  # Maximum tastiness when upset

    for x, y in courses:
        if x == 0:  # Antidotal course
            healthy = max(healthy, upset + y)  # If we eat, we can become healthy
        else:  # Poisonous course
            upset = max(upset, healthy + y)  # If we eat, we can become upset

    return max(healthy, 0)  # Return the maximum tastiness or 0 if nothing is eaten

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
courses = [tuple(map(int, line.split())) for line in data[1:N+1]]

print(max_tastiness(N, courses))