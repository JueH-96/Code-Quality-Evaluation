import sys
import math

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Read Q hubs
hubs = []
for _ in range(Q):
    L = int(data[index])
    index += 1
    R = int(data[index])
    index += 1
    C = int(data[index])
    index += 1
    hubs.append((C, L, R))

# Sort hubs by cost ascending
hubs.sort()

# Initialize intervals for original vertices, each as a single point
intervals = [(i, i) for i in range(1, N + 1)]

# Initialize component count and total cost
comp_count = N + Q
total_cost = 0

# Binary search to find largest index with start <= B
def find_P(intervals, B):
    left, right = 0, len(intervals) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if intervals[mid][0] <= B:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result

# Binary search to find smallest index with end >= A
def find_Q(intervals, A):
    left, right = 0, len(intervals) - 1
    result = len(intervals)
    while left <= right:
        mid = (left + right) // 2
        if intervals[mid][1] >= A:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result

# Binary search to find insertion index for start_new
def find_insert_idx(intervals, start_new):
    left, right = 0, len(intervals)
    while left < right:
        mid = (left + right) // 2
        if intervals[mid][0] < start_new:
            left = mid + 1
        else:
            right = mid
    return left

# Process each hub in sorted order
for cost, L, R in hubs:
    # Find Q_idx and P_idx
    Q_idx = find_Q(intervals, L)
    P_idx = find_P(intervals, R)
    if Q_idx <= P_idx and P_idx >= 0 and Q_idx < len(intervals):
        k = P_idx - Q_idx + 1
    else:
        k = 0  # Should not happen, but handle safely
    
    # Add to total cost
    total_cost += k * cost
    
    # Decrease component count
    comp_count -= k
    
    # If k > 0, remove the intervals and add new one
    if k > 0:
        start_min = intervals[Q_idx][0]
        end_max = intervals[P_idx][1]
        # Remove the slice
        del intervals[Q_idx : P_idx + 1]
        # Find insertion index and insert new interval
        idx_insert = find_insert_idx(intervals, start_min)
        intervals.insert(idx_insert, (start_min, end_max))

# After processing all hubs, check connectivity
if comp_count == 1:
    print(total_cost)
else:
    print(-1)