def min_max_diff(N, K, A):
    # Sort the array to find the differences between consecutive elements
    A.sort()
    # Calculate the differences between consecutive elements
    diffs = [A[i+1] - A[i] for i in range(N-1)]
    # Sort the differences
    diffs.sort()
    # The strategy is to remove K largest differences to minimize the range
    # Since we can remove K elements, we can remove K-1 differences
    # The minimum possible value is the sum of the smallest N-K-1 differences
    return sum(diffs[:N-K-1])

# Read input from stdin
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Get the result
result = min_max_diff(N, K, A)

# Write the result to stdout
print(result)