import sys

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Sort the transportation costs in ascending order
A.sort()

# Initialize the maximum subsidy limit
max_subsidy = 0

# Iterate through the sorted transportation costs
total_subsidy = 0
for a in A:
    # Calculate the current subsidy limit
    subsidy_limit = min(a, M - total_subsidy)
    
    # Update the maximum subsidy limit
    max_subsidy = max(max_subsidy, subsidy_limit)
    
    # Update the total subsidy
    total_subsidy += subsidy_limit
    
    # If the total subsidy exceeds the budget, break the loop
    if total_subsidy > M:
        break

# If the total subsidy is less than or equal to the budget, the subsidy limit can be made infinitely large
if total_subsidy <= M:
    print("infinite")
else:
    print(max_subsidy)