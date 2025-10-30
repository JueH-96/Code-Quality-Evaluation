import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
S = data[2]
X = list(map(int, data[3:]))

# Create a list of tuples (position, direction, index)
ants = [(X[i], 1 if S[i] == '1' else -1, i) for i in range(N)]

# Sort ants by their initial positions
ants.sort()

# Dictionary to store the events
events = defaultdict(list)

# Check for each pair of ants if they will meet before time T
for i in range(N):
    for j in range(i + 1, N):
        if ants[i][1] == 1 and ants[j][1] == -1:
            # Calculate the time they will meet
            time_meet = (ants[j][0] - ants[i][0]) / 2.0
            if time_meet <= T:
                events[time_meet].append((ants[i][2], ants[j][2]))

# Count the number of pairs that meet before time T
count = 0
for time_meet in sorted(events):
    for pair in events[time_meet]:
        count += 1

print(count)