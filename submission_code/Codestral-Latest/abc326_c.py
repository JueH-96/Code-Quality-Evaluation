import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Sort the gift coordinates
A.sort()

# Use a sliding window approach to find the maximum number of gifts in any interval of length M
max_gifts = 0
left = 0

for right in range(N):
    while A[right] - A[left] >= M:
        left += 1
    max_gifts = max(max_gifts, right - left + 1)

print(max_gifts)