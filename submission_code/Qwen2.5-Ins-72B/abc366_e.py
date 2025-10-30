# YOUR CODE HERE
import sys
from collections import defaultdict

def solve(N, D, points):
    # Precompute the sums and differences of the points
    sums = defaultdict(int)
    diffs = defaultdict(int)
    for x, y in points:
        sums[x + y] += 1
        diffs[x - y] += 1

    # Calculate the prefix sums for sums and diffs
    sum_keys = sorted(sums.keys())
    diff_keys = sorted(diffs.keys())
    sum_prefix = [0] * (len(sum_keys) + 1)
    diff_prefix = [0] * (len(diff_keys) + 1)
    for i, key in enumerate(sum_keys):
        sum_prefix[i + 1] = sum_prefix[i] + sums[key]
    for i, key in enumerate(diff_keys):
        diff_prefix[i + 1] = diff_prefix[i] + diffs[key]

    # Function to count the number of valid points for a given sum or diff
    def count_valid_points(prefix, keys, value, D):
        left = 0
        right = len(keys) - 1
        while left <= right:
            mid = (left + right) // 2
            if keys[mid] <= value + D:
                left = mid + 1
            else:
                right = mid - 1
        return prefix[left]

    # Calculate the total number of valid points
    total = 0
    for x in range(-10**6, 10**6 + 1):
        for y in range(-10**6, 10**6 + 1):
            if sum(count_valid_points(sum_prefix, sum_keys, x + y, D) for _ in range(2)) + sum(count_valid_points(diff_prefix, diff_keys, x - y, D) for _ in range(2)) <= D:
                total += 1

    return total

# Read input
input = sys.stdin.read
N, D, *points = map(int, input().split())
points = list(zip(points[::2], points[1::2]))

# Solve and print the result
print(solve(N, D, points))