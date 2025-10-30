import sys

def calculate_subsidy(A, x):
    """Calculate the total subsidy for a given subsidy limit x."""
    return sum(min(a, x) for a in A)

def find_max_subsidy_limit(N, M, A):
    """Find the maximum possible value of the subsidy limit x."""
    A.sort()  # Sort the transportation costs in ascending order

    # Check if the subsidy limit can be made infinitely large
    if sum(A) <= M:
        return "infinite"

    # Initialize the search range for the subsidy limit
    low = 0
    high = max(A)

    # Perform binary search to find the maximum subsidy limit
    while low < high:
        mid = (low + high + 1) // 2  # +1 to round up for the mid value
        if calculate_subsidy(A, mid) <= M:
            low = mid
        else:
            high = mid - 1

    return low

def main():
    # Read input from stdin
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Find and print the maximum possible value of the subsidy limit
    max_subsidy_limit = find_max_subsidy_limit(N, M, A)
    print(max_subsidy_limit)

if __name__ == "__main__":
    main()