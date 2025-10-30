class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        from typing import List
        
        # Fenwick Tree (Binary Indexed Tree) implementation
        class FenwickTree:
            def __init__(self, n):
                self.n = n
                self.fw = [0] * (n + 1)
            
            # Update Fenwick tree at index i (0-based) by delta
            def update(self, i, delta):
                i += 1  # switch to 1-based index
                while i <= self.n:
                    self.fw[i] += delta
                    i += i & -i
            
            # Query Fenwick tree prefix sum [0..i-1] (0-based)
            def query(self, i):
                s = 0
                while i > 0:
                    s += self.fw[i]
                    i -= i & -i
                return s
            
            # Get sum in [l..r] (0-based)
            def range_sum(self, l, r):
                if r < l:
                    return 0
                return self.query(r + 1) - self.query(l)
        
        def is_peak(arr, i):
            # Return 1 if arr[i] is a peak, else 0
            if 0 < i < len(arr) - 1:
                return 1 if (arr[i] > arr[i - 1] and arr[i] > arr[i + 1]) else 0
            return 0
        
        n = len(nums)
        
        # Build initial peak array p
        p = [0] * n
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                p[i] = 1
        
        # Build Fenwick tree based on p
        fenw = FenwickTree(n)
        for i in range(n):
            if p[i] == 1:
                fenw.update(i, 1)
        
        ans = []
        
        for q in queries:
            if q[0] == 1:
                # Query type 1: [1, l, r] -> number of peaks in subarray [l..r]
                l, r = q[1], q[2]
                if r - l < 2:
                    ans.append(0)
                else:
                    # count peaks in [l+1 .. r-1]
                    ans.append(fenw.range_sum(l + 1, r - 1))
            else:
                # Query type 2: [2, index, val] -> update nums[index] = val
                idx, val = q[1], q[2]
                nums[idx] = val
                # Recompute peak values for idx-1, idx, idx+1
                for i in (idx - 1, idx, idx + 1):
                    if 0 <= i < n:
                        new_val = is_peak(nums, i)
                        if new_val != p[i]:
                            fenw.update(i, new_val - p[i])
                            p[i] = new_val
        
        return ans