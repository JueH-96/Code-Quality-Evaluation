class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        """
        We will maintain a Fenwick Tree (Binary Indexed Tree) to track peak counts.
        Define an array peak_status where peak_status[i] = 1 if nums[i] is a "peak",
        and 0 otherwise. A peak is defined only for 0 < i < n-1 and nums[i] > nums[i-1]
        and nums[i] > nums[i+1].

        Then, for queries of type [1, l, r], we will return the sum of peak_status
        in the range (l+1) to (r-1), since the problem states that the first and last
        element in the subarray cannot be considered a peak.

        For queries of type [2, index, val], we update nums[index] = val, and then
        recalculate and update the Fenwick tree for index-1, index, and index+1
        (whenever those indices are valid for a peak).
        """

        # Fenwick (Binary Indexed) Tree implementation
        class FenwickTree:
            def __init__(self, n):
                self.n = n
                self.tree = [0] * (n + 1)

            def update(self, i, delta):
                """Adds 'delta' to element with index 'i' (1-indexed)."""
                while i <= self.n:
                    self.tree[i] += delta
                    i += i & -i

            def query(self, i):
                """Returns the prefix sum from 1 to i (inclusive)."""
                s = 0
                while i > 0:
                    s += self.tree[i]
                    i -= i & -i
                return s

            def range_sum(self, left, right):
                """Returns the sum in the interval [left, right], 1-indexed."""
                if left > right:
                    return 0
                return self.query(right) - self.query(left - 1)

        n = len(nums)
        # A helper function to check if nums[i] is a peak (valid for 0 < i < n-1).
        def is_peak(arr, i):
            return 0 < i < n-1 and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]

        # Build initial peak_status array
        peak_status = [0] * n
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                peak_status[i] = 1

        # Fenwick tree on peak_status
        fenw = FenwickTree(n)
        # Build fenw with initial values
        for i in range(1, n - 1):
            if peak_status[i] == 1:
                fenw.update(i + 1, 1)  # 1-based index in Fenwick

        ans = []
        for q in queries:
            if q[0] == 1:
                # Query type [1, l, r]: count the number of peaks in subarray [l..r]
                l, r = q[1], q[2]
                # The peaks can only lie from (l+1) to (r-1)
                if l + 1 <= r - 1:
                    left_idx = l + 2  # Fenw is 1-based, so i+1 => (l+1)+1
                    right_idx = r     # (r-1)+1
                    count_peaks = fenw.range_sum(left_idx, right_idx)
                else:
                    count_peaks = 0
                ans.append(count_peaks)

            else:
                # Query type [2, index, val]: update nums[index] = val
                index, val = q[1], q[2]
                nums[index] = val

                # We need to potentially update peak_status at index-1, index, index+1
                for i in [index - 1, index, index + 1]:
                    if 0 <= i < n:
                        old_val = peak_status[i]
                        new_val = 1 if is_peak(nums, i) else 0
                        if old_val != new_val:
                            # Update Fenwicks tree
                            fenw.update(i + 1, new_val - old_val)  # 1-based indexing
                            peak_status[i] = new_val

        return ans