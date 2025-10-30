import sys
import bisect

# Read all input and convert to integers
data = list(map(int, sys.stdin.read().split()))

# Parse N and M
index = 0
N = data[index]
index += 1
M = data[index]
index += 1

# Read A and B lists
A = data[index:index + N]
index += N
B = data[index:index + M]

# Sort the lists
A_sorted = sorted(A)
B_sorted = sorted(B)

# Binary search to find the minimum X
low = 1
high = 1000000000  # 10^9

while low <= high:
    mid = (low + high) // 2
    # Number of sellers who can sell at mid or less
    s_mid = bisect.bisect_right(A_sorted, mid)
    # Number of buyers who can buy at mid or more (B_i >= mid)
    b_mid = len(B_sorted) - bisect.bisect_left(B_sorted, mid)
    if s_mid >= b_mid:
        high = mid - 1
    else:
        low = mid + 1

# After binary search, low is the minimum X that satisfies the condition
print(low)