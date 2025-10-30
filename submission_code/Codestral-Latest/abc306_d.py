import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
courses = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(n)]

# Initialize variables to track the maximum tastiness for healthy and upset stomach states
max_tastiness_healthy = 0
max_tastiness_upset = 0

for x, y in courses:
    if x == 0:  # Antidotal course
        max_tastiness_healthy = max(max_tastiness_healthy, max_tastiness_upset + y)
    else:  # Poisonous course
        max_tastiness_upset = max(max_tastiness_upset, max_tastiness_healthy + y)

# The answer is the maximum tastiness when Takahashi is healthy after eating all courses
print(max_tastiness_healthy)