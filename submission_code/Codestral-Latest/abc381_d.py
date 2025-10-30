# YOUR CODE HERE
import sys
from collections import defaultdict

def max_1122_sequence_length(N, A):
    max_length = 0
    count = defaultdict(int)

    left = 0
    for right in range(N):
        count[A[right]] += 1

        while any(count[val] > 2 for val in count) or (right - left + 1) % 2 != 0:
            count[A[left]] -= 1
            if count[A[left]] == 0:
                del count[A[left]]
            left += 1

        if (right - left + 1) % 2 == 0:
            max_length = max(max_length, right - left + 1)

    return max_length

# Read input
N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

# Calculate and print the result
result = max_1122_sequence_length(N, A)
print(result)