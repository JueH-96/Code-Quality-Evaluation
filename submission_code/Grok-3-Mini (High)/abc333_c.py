# YOUR CODE HERE
import sys
data = sys.stdin.read().strip()
N = int(data)
# Generate all possible sums of three repunits with a <= b <= c <= 12
sums_list = []
for c in range(1, 13):  # c from 1 to 12
    for b in range(1, c + 1):  # b from 1 to c
        for a in range(1, b + 1):  # a from 1 to b
            T = (10 ** a) + (10 ** b) + (10 ** c)
            S = (T - 3) // 9
            sums_list.append(S)
# Sort the list in ascending order
sums_list.sort()
# The N-th smallest is at index N-1 (0-based indexing)
answer = sums_list[N - 1]
# Print the answer
print(answer)