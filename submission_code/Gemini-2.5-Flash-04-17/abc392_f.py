import sys

# Function to implement Binary Indexed Tree (Fenwick Tree)
class FenwickTree:
    def __init__(self, size):
        self.size = size
        # tree is 1-indexed internally
        self.tree = [0] * (size + 1)

    # Update the value at index idx by delta
    # idx is 0-indexed externally, converted to 1-indexed internally
    def update(self, idx, delta):
        idx += 1  # Convert to 1-based indexing
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & (-idx)

    # Get the prefix sum up to index idx (inclusive)
    # idx is 0-indexed externally, converted to 1-indexed internally
    def query(self, idx):
        idx += 1  # Convert to 1-based indexing
        prefix_sum = 0
        while idx > 0:
            prefix_sum += self.tree[idx]
            idx -= idx & (-idx)
        return prefix_sum

    # Find the smallest 0-indexed position `idx` such that `query(idx)`
    # (number of available slots in [0, idx]) is >= k (1-based rank).
    # This uses binary search and query, resulting in O(log^2 N) time.
    # It finds the first index where the cumulative count reaches or exceeds k.
    def find_kth(self, k):
        # Binary search for the 0-indexed position
        low = 0
        high = self.size - 1
        result_idx = -1  # Store the potential answer

        while low <= high:
            mid = low + (high - low) // 2
            # Check the number of available slots up to `mid` (0-indexed)
            if self.query(mid) >= k:
                # mid is a possible answer, try finding an earlier one
                result_idx = mid
                high = mid - 1
            else:
                # mid has too few available slots, search in the right half
                low = mid + 1
                
        # result_idx will be the smallest 0-indexed index whose prefix sum (count of available slots) is >= k.
        # This corresponds to the k-th available slot.
        return result_idx

# Main logic
def solve():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    # The final array has N elements. We will determine their final positions
    # by considering the insertions in reverse order from N down to 1.
    # We use a BIT to track which positions in the final array are still "available"
    # to be filled by the numbers we are currently considering (from N down to 1).
    # Initially, all N slots (indices 0 to N-1) in the final array are available.
    # The BIT will store the count of available slots.

    ft = FenwickTree(N)
    # Mark all slots as available initially
    for i in range(N):
        ft.update(i, 1) 

    result = [0] * N # This will store the final array

    # Iterate from i = N down to 1
    # When we consider number i, it was the i-th number inserted.
    # At that time, there were i-1 numbers already in the list, occupying i-1 positions.
    # The number i was inserted at the P[i-1]-th position (1-based) relative to the list of size i-1.
    # In the final array of N slots, the numbers from 1 to i will occupy i slots.
    # The number i will be placed into the P[i-1]-th available slot among the N slots.
    # The number of available slots decreases as we place numbers from N down to 1.
    # When placing number i, there are exactly i available slots in the final array
    # that are not yet filled by numbers > i.
    
    for i in range(N, 0, -1):
        # P is 0-indexed from input. P_i (1-based in problem) corresponds to P[i-1] (0-indexed).
        # The rank is the target position (1-based) among the available slots.
        rank = P[i-1] 

        # Find the index (0-based) of the rank-th available slot in the final array.
        # The BIT stores counts of available slots.
        slot_index = ft.find_kth(rank)

        # Place number i into this found slot index.
        result[slot_index] = i

        # Mark this slot as occupied (no longer available for numbers less than i)
        ft.update(slot_index, -1)

    # Output the final array elements separated by spaces
    print(*result)

solve()