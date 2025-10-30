import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Dictionary to store the count of differences for each index
diff_count = defaultdict(int)

# Dictionary to store the count of arithmetic progressions ending at each index
ap_count = [0] * N

# Calculate the number of arithmetic progressions ending at each index
for i in range(1, N):
    diff = A[i] - A[i - 1]
    ap_count[i] = ap_count[i - 1] + diff_count[diff]
    diff_count[diff] += 1

# Calculate the total number of arithmetic progressions
total_ap = N  # Each single element is an arithmetic progression
for i in range(1, N):
    total_ap += ap_count[i]

print(total_ap)