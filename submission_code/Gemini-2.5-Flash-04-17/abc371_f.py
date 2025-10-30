import sys
import heapq

# Custom class to hold heap with sum
class SumHeap:
    def __init__(self, is_min_heap):
        self.heap = []
        self.is_min_heap = is_min_heap
        self.sum = 0
        self.size = 0

    def push(self, val):
        self.sum += val
        self.size += 1
        if self.is_min_heap:
            heapq.heappush(self.heap, val)
        else:
            heapq.heappush(self.heap, -val)

    def pop(self):
        if not self.heap: return None
        val = heapq.heappop(self.heap)
        if not self.is_min_heap: val = -val
        self.sum -= val
        self.size -= 1
        return val

    def top(self):
        if not self.heap: return None
        val = self.heap[0]
        if not self.is_min_heap: val = -val
        return val

    def __len__(self):
        return self.size

# Helper to rebalance heaps
def rebalance_heaps(min_heap, max_heap):
    while len(max_heap) > len(min_heap) + 1:
        min_heap.push(max_heap.pop())
    while len(min_heap) > len(max_heap):
        max_heap.push(min_heap.pop())

# Helper to merge heaps
def merge_sum_heaps(min1, max1, min2, max2):
    # Transfer elements from smaller heaps to larger
    if len(min1) + len(max1) < len(min2) + len(max2):
        min1, max1, min2, max2 = min2, max2, min1, max1 # Ensure (min1, max1) is larger

    while min2.size > 0: min1.push(min2.pop())
    while max2.size > 0: max1.push(max2.pop())

    rebalance_heaps(min1, max1)
    return min1, max1

# PAVA function using stack and heaps for L1 with fixed point
# Input: list of values a, target_idx, target_val
# Output: list of smoothed values
def apply_pava_fixed(a, target_idx, target_val):
    n = len(a)
    if n == 0: return []

    stack = [] # Stores blocks: (start_idx, end_idx, min_heap, max_heap)

    for i in range(n):
        val = a[i]
        if i == target_idx:
            val = target_val

        # New block for val
        current_min_heap = SumHeap(True)
        current_max_heap = SumHeap(False)
        current_max_heap.push(val) # Initially push to max_heap
        rebalance_heaps(current_min_heap, current_max_heap)

        current_block = (i, i, current_min_heap, current_max_heap)
        
        # Merge with previous blocks if median condition is violated
        while stack:
            last_block = stack[-1]
            # Check median condition: median of last block <= median of current block
            if last_block[2].top() > current_block[2].top(): # last median > current median
                stack.pop()
                # Merge heaps
                merged_min_heap, merged_max_heap = merge_sum_heaps(last_block[2], last_block[3], current_block[2], current_block[3])
                current_block = (last_block[0], current_block[1], merged_min_heap, merged_max_heap)
            else:
                break # Condition met

        stack.append(current_block)

    # Build the resulting smoothed sequence
    result = [0] * n
    for block in stack:
        start, end, min_h, max_h = block
        median_val = max_h.top() # Median is the largest in the lower half (max_heap)
        for j in range(start, end + 1):
            result[j] = median_val

    return result

# Main part of the solution
def solve():
    N = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())
    Tasks = []
    for _ in range(Q):
        T, G = map(int, sys.stdin.readline().split())
        Tasks.append((T, G))

    # Initial y values
    # y_i = P_i + i
    # 0-indexed: y[i] = P[i] + (i + 1)
    y = [X[i] + (i + 1) for i in range(N)]

    total_movement = 0

    for task in Tasks:
        T, G = task # T is 1-based index

        # The problem asks for minimum total movement sum |P'_j - P_j|.
        # With y_j = P_j + j, this is sum |(y'_j - j) - (y_j - j)| = sum |y'_j - y_j|.
        # We need y'_1 < y'_2 < ... < y'_N and y'_T = G+T.
        # This is L1 isotonic regression with a point constraint.
        # The optimal sequence y' is obtained by applying PAVA on the sequence y,
        # but fixing the T-th element (at index T-1) to G+T.
        
        y_prime = apply_pava_fixed(y, target_idx=T-1, target_val=G + T)

        # The cost for this task is the sum of absolute differences between the new y and old y.
        task_cost = 0
        for j in range(N):
            task_cost += abs(y_prime[j] - y[j])

        total_movement += task_cost

        # Update y for the next task
        y = y_prime

    print(total_movement)

solve()