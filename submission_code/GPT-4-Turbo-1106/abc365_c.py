# Read the inputs
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Sort the transportation costs in ascending order
A.sort()

# Initialize the subsidy limit to 0
x = 0

# Check if the subsidy limit can be made infinitely large
if sum(A) <= M:
    print("infinite")
else:
    # Calculate the maximum possible value of the subsidy limit
    total_subsidy = 0
    for cost in A:
        if total_subsidy + cost <= M:
            total_subsidy += cost
            x = cost
        else:
            x = M - (total_subsidy - cost)
            break
    print(x)