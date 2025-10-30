import sys

# Read all input from stdin
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Read W and X for each base and store in a list of tuples
bases = []
for _ in range(N):
    W = int(data[index])
    X = int(data[index + 1])
    bases.append((W, X))
    index += 2

# Initialize maximum sum
max_sum = 0

# Iterate over all possible UTC start times S from 0 to 23
for S in range(24):
    current_sum = 0
    # For each base, check if they can participate at time S
    for W, X in bases:
        local_time = (S + X) % 24
        if 9 <= local_time <= 17:
            current_sum += W
    # Update maximum sum if current sum is larger
    if current_sum > max_sum:
        max_sum = current_sum

# Output the maximum sum
print(max_sum)