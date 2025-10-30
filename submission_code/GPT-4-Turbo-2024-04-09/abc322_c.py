import sys
import bisect

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# We will use binary search to find the first day with fireworks on or after each day i
results = []
for i in range(1, N + 1):
    # Find the index of the first day in A that has fireworks on or after day i
    idx = bisect.bisect_left(A, i)
    if idx < M:
        # Calculate the number of days until the next fireworks
        results.append(A[idx] - i)
    else:
        # If no fireworks day is found, it should not happen as A_M = N is guaranteed
        results.append(0)

# Print the results
for result in results:
    print(result)