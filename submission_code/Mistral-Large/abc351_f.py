import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Initialize variables
total_sum = 0
current_sum = 0
max_diff = 0

# Create a list to store the differences
diff = []

# Calculate the differences and store them in the list
for i in range(1, N):
    diff.append(A[i] - A[i-1])

# Iterate through the differences list
for d in diff:
    if d > 0:
        current_sum += d
        max_diff = max(max_diff, current_sum)
    else:
        current_sum = 0

# Add the maximum difference to the total sum
total_sum += max_diff

# Print the result
print(total_sum)