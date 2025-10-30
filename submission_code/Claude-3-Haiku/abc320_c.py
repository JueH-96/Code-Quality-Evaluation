import sys

# Read input
M = int(input())
S1 = input()
S2 = input()
S3 = input()

# Initialize the minimum time to stop all reels
min_time = float('inf')

# Try all possible times to stop the reels
for t in range(M):
    # Check if all reels can be stopped at time t
    if (t + 1) == int(S1[t % M]) == int(S2[t % M]) == int(S3[t % M]):
        min_time = min(min_time, t + 1)

# If it's impossible to stop all reels, print -1
if min_time == float('inf'):
    print(-1)
else:
    print(min_time)