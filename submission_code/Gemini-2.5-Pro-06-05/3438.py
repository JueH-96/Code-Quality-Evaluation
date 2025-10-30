from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # Fenwick Tree (Binary Indexed Tree) using 1-based indexing for its internal array.
        # The BIT will store a 1 for each index that is a peak, and 0 otherwise.
        bit = [0] * (n + 1)

        def update(i: int, delta: int):
            """Updates the BIT by adding delta to index i."""
            # i is the 0-based index from the original nums array.
            # We convert it to 1-based for the BIT logic.
            i += 1
            while i <= n:
                bit[i] += delta
                i += i & -i

        def query(i: int) -> int:
            """Queries the prefix sum up to index i (inclusive)."""
            # i is the 0-based index.
            if i < 0:
                return 0
            s = 0
            i += 1
            while i > 0:
                s += bit[i]
                i -= i & -i
            return s
        
        def range_query(l: int, r: int) -> int:
            """Queries the sum in the range [l, r] (inclusive)."""
            if l > r:
                return 0
            return query(r) - query(l - 1)

        def is_peak(i: int) -> bool:
            """Checks if the element at index i is a peak."""
            # By definition, a peak cannot be the first or last element of the main array.
            # So, we only check for 0 < i < n - 1.
            if not (0 < i < n - 1):
                return False
            return nums[i] > nums[i-1] and nums[i] > nums[i+1]

        # Initial population of the Fenwick Tree
        for i in range(1, n - 1):
            if is_peak(i):
                update(i, 1)

        ans = []
        for q_type, arg1, arg2 in queries:
            if q_type == 1:
                # Type 1 query: [1, l, r] - count peaks in nums[l..r]
                l, r = arg1, arg2
                # A peak in a subarray cannot be its first or last element.
                # This means we are looking for peaks at indices i such that l < i < r.
                # So, the query range is [l + 1, r - 1].
                if r - l < 2:
                    ans.append(0)
                else:
                    count = range_query(l + 1, r - 1)
                    ans.append(count)
            
            else: # q_type == 2
                # Type 2 query: [2, index, val] - update nums[index] to val
                index, val = arg1, arg2

                if nums[index] == val:
                    continue

                # An update at `index` can affect the peak status of `index-1`, `index`, and `index+1`.
                # We need to re-evaluate the peak status for these indices.
                # We only care about indices that can be peaks, i.e., those in [1, n-2].
                indices_to_recalc = set()
                for i in range(max(1, index - 1), min(n - 1, index + 2)):
                    indices_to_recalc.add(i)
                
                # Store the old peak statuses for the affected indices before the update.
                old_statuses = {i: 1 for i in indices_to_recalc if is_peak(i)}
                
                # Perform the update on the nums array.
                nums[index] = val
                
                # Recalculate peak statuses for affected indices and update the BIT.
                for i in indices_to_recalc:
                    old_status = old_statuses.get(i, 0)
                    new_status = 1 if is_peak(i) else 0
                    delta = new_status - old_status
                    if delta != 0:
                        update(i, delta)

        return ans