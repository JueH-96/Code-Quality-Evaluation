import sys

def max_subsidy_limit(N, M, A):
    # Sort the transportation costs
    A.sort()

    # Initialize the subsidy limit
    x = 0

    # Check if the sum of all A_i is less than or equal to M
    if sum(A) <= M:
        return "infinite"

    # Binary search for the maximum subsidy limit
    left, right = 0, max(A)
    while left <= right:
        mid = (left + right) // 2
        total_subsidy = sum(min(mid, a) for a in A)

        if total_subsidy <= M:
            x = mid
            left = mid + 1
        else:
            right = mid - 1

    return x

# Read input from stdin
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Calculate the maximum subsidy limit
result = max_subsidy_limit(N, M, A)

# Write the output to stdout
sys.stdout.write(str(result) + '
')