from typing import List

class FenwickTree:
    def __init__(self, n):
        self.size = n
        # Use 1-based indexing internally for easier bit manipulation
        self.tree = [0] * (n + 1)

    def update(self, idx, delta):
        # Add delta to the element at 0-based index idx
        idx += 1 # Convert to 1-based index
        while idx <= self.size:
            self.tree[idx] += delta
            # Move to the next relevant node
            idx += idx & (-idx)

    def query(self, idx):
        # Get prefix sum up to 0-based index idx (inclusive)
        idx += 1 # Convert to 1-based index
        total_sum = 0
        while idx > 0:
            total_sum += self.tree[idx]
            # Move to the parent node
            idx -= idx & (-idx)
        return total_sum

    def query_range(self, l, r):
        # Get sum for 0-based index range [l, r] inclusive
        if l > r:
            return 0
        # Sum [l...r] = Sum [0...r] - Sum [0...l-1]
        # query(l-1) is handled correctly by query() returning 0 for idx=-1 (input)
        return self.query(r) - self.query(l - 1)


class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # Fenwick tree of size n to cover indices 0 to n-1
        ft = FenwickTree(n)

        # Helper function to check if index i is a peak in the current nums array state
        def is_peak(i, arr):
            N = len(arr)
            # An index i can be a peak only if it's not the first or last element (0 < i < N-1)
            # and it's greater than both its neighbors.
            if 0 < i < N - 1 and arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                return True
            return False

        # Initial build of Fenwick tree: add 1 for each initial peak
        # Only indices 1 to n-2 can potentially be peaks in the original array
        for i in range(1, n - 1):
            if is_peak(i, nums):
                ft.update(i, 1)

        results = []

        for query in queries:
            type = query[0]
            if type == 1:
                # Query [1, l, r]: count peaks in the subarray nums[l..r]
                l, r = query[1], query[2]
                # According to the note, the first (index l) and last (index r)
                # elements of the subarray cannot be counted as peaks.
                # We need to count peaks at indices i such that l < i < r.
                # This means we count peaks at original indices from l+1 to r-1.
                # The peak check itself is always done against the original array `nums`.
                count = ft.query_range(l + 1, r - 1)
                results.append(count)
            else: # type == 2
                # Query [2, index, val]: change nums[index] to val
                index, val = query[1], query[2]

                # Identify indices whose peak status might change due to the update.
                # These are the index itself and its immediate neighbors: index-1, index, index+1.
                # We only care about indices that are potential peaks in the original array (0 < i < n-1).
                indices_to_recheck = set()
                if index > 0: indices_to_recheck.add(index - 1)
                indices_to_recheck.add(index)
                if index < n - 1: indices_to_recheck.add(index + 1)

                # Before updating nums:
                # For each index that might be affected, if it was a peak,
                # decrement its count in the Fenwick tree.
                for i in indices_to_recheck:
                    # The is_peak function correctly handles indices 0 and n-1
                    # by returning False if 0 < i < n-1 is not met, so we
                    # only decrement if it was truly a peak at a valid position.
                    if is_peak(i, nums):
                        ft.update(i, -1)

                # Perform the update
                nums[index] = val

                # After updating nums:
                # For each index that might have been affected, if it is now a peak,
                # increment its count in the Fenwick tree.
                for i in indices_to_recheck:
                     if is_peak(i, nums):
                        ft.update(i, 1)

        return results