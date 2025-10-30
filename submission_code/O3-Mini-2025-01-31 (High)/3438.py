from typing import List

# We'll use a Binary Indexed Tree (Fenwick Tree) to support efficient point updates and range sum queries.
class BIT:
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)
        
    # update BIT at position i (1-indexed) with delta
    def update(self, i: int, delta: int):
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i
            
    # query returns cumulative sum from index 1 to i (inclusive)
    def query(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # BIT is used over the indices of the array.
        # We only consider indices 1 through n-2 as potential peaks since
        # the first and last elements can never be peaks.
        bit = BIT(n)
        # is_peak[i] indicates if nums[i] is a peak (1 if yes, 0 if not)
        is_peak = [0] * n
        
        # Initialize peak information for indices 1 to n-2.
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                is_peak[i] = 1
                # In BIT, we use 1-indexing so update at position i+1.
                bit.update(i + 1, 1)
                
        res = []
        
        # Helper function: for an index i that is in the valid range [1, n-2],
        # recalc if nums[i] is a peak and update the BIT if its status changes.
        def update_candidate(i: int):
            if 1 <= i <= n - 2:
                new_val = 1 if nums[i] > nums[i - 1] and nums[i] > nums[i + 1] else 0
                if new_val != is_peak[i]:
                    diff = new_val - is_peak[i]
                    bit.update(i + 1, diff)
                    is_peak[i] = new_val
                    
        # Process all queries.
        for query in queries:
            if query[0] == 1:
                # Query type 1: Count peaks in subarray nums[l...r].
                # Recall: the first and the last element of a subarray cannot be peaks,
                # so we count peaks only for indices in the range [l+1, r-1].
                l, r = query[1], query[2]
                if r - l < 2:
                    res.append(0)
                else:
                    # In BIT, an element at original index i is stored at position i+1.
                    # So to sum candidate indices l+1 through r-1:
                    #   sum = BIT.query((r - 1) + 1) - BIT.query((l + 1))
                    count = bit.query(r) - bit.query(l + 1)
                    res.append(count)
            else:
                # Query type 2: Update query.  [2, index, val]
                pos, val = query[1], query[2]
                # Update the value in the original array.
                nums[pos] = val
                # The update at pos might affect
                # the peak status of pos-1, pos, and pos+1 (if they're in the valid range).
                for i in (pos - 1, pos, pos + 1):
                    if 1 <= i <= n - 2:
                        update_candidate(i)
        return res

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    nums1 = [3, 1, 4, 2, 5]
    queries1 = [[2, 3, 4], [1, 0, 4]]
    print(sol.countOfPeaks(nums1, queries1))  # Output: [0]
    
    # Example 2:
    nums2 = [4, 1, 4, 2, 1, 5]
    queries2 = [[2, 2, 4], [1, 0, 2], [1, 0, 4]]
    print(sol.countOfPeaks(nums2, queries2))  # Output: [0, 1]