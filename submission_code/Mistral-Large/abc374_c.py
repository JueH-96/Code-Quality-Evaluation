import sys

def minimize_max_group_size(N, K):
    # Sort the list of department sizes
    K.sort()

    # Initialize the left and right pointers for the binary search
    left, right = max(K), sum(K)

    while left < right:
        mid = (left + right) // 2
        # Check if it's possible to divide the departments into two groups
        # such that the maximum group size is at most `mid`
        current_sum = 0
        max_group_size = 0
        for size in K:
            if current_sum + size > mid:
                max_group_size = max(max_group_size, current_sum)
                current_sum = size
            else:
                current_sum += size
        max_group_size = max(max_group_size, current_sum)

        if max_group_size <= mid:
            right = mid
        else:
            left = mid + 1

    return left

# Read input from stdin
input = sys.stdin.read
data = input().split()
N = int(data[0])
K = list(map(int, data[1:]))

# Calculate the result
result = minimize_max_group_size(N, K)

# Write the output to stdout
print(result)