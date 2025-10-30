import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Extract N and M
N = int(data[0])
M = int(data[1])

# Extract the required amounts A for each nutrient
A = [int(data[i]) for i in range(2, 2 + M)]

# Calculate the starting index for the X values
start_X = 2 + M

# Flag to check if all nutrient goals are met
all_met = True

# Check each nutrient
for j in range(M):
    # Calculate the total intake for nutrient j
    total = sum(int(data[start_X + i * M + j]) for i in range(N))
    # If the total is less than required, set flag to False and break
    if total < A[j]:
        all_met = False
        break

# Output the result
if all_met:
    print("Yes")
else:
    print("No")