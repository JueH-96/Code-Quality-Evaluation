import sys

# Read the number of weeks
N = int(sys.stdin.readline().strip())

# Read the steps for each day
A = list(map(int, sys.stdin.readline().strip().split()))

# Initialize the result list
B = []

# Calculate the sum of steps for each week
for i in range(N):
    week_sum = sum(A[i*7:(i+1)*7])
    B.append(week_sum)

# Print the result
print(' '.join(map(str, B)))