import sys

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + M]))

# Sort both arrays in ascending order
A.sort()
B.sort()

# Initialize pointer and total cost
ptr = 0
total_cost = 0
possible = True

# Iterate through each B value
for b_val in B:
    # Skip all A values that are less than b_val
    while ptr < N and A[ptr] < b_val:
        ptr += 1
    # Check if there is a suitable A value
    if ptr < N:
        # Assign the A value and add to cost
        total_cost += A[ptr]
        ptr += 1
    else:
        # No suitable A found, impossible
        possible = False
        break

# Output the result
if not possible:
    print(-1)
else:
    print(total_cost)