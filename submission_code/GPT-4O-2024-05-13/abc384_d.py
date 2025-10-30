# YOUR CODE HERE
def find_subsequence_sum(N, S, A):
    # Use a sliding window approach to find the subsequence sum
    current_sum = 0
    start = 0
    for end in range(2 * N):  # We only need to check up to 2N elements
        current_sum += A[end % N]
        while current_sum > S and start <= end:
            current_sum -= A[start % N]
            start += 1
        if current_sum == S:
            return "Yes"
    return "No"

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = int(data[1])
A = list(map(int, data[2:]))

print(find_subsequence_sum(N, S, A))