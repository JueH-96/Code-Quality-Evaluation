import sys
from collections import Counter

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Count the frequency of each element in A
counter = Counter(A)

# Find the maximum size of the Pyramid Sequence
max_size = 0
for k in range(1, N + 1):
    # Check if a Pyramid Sequence of size k can be formed
    if all(counter[i] >= 2 * (k - i) for i in range(1, k)):
        max_size = k
    else:
        break

print(max_size)