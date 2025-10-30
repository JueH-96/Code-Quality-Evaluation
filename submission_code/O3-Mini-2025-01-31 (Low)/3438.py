class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # We'll use Fenwick Tree (Binary Indexed Tree) to support point updates and range queries.
        n = len(nums)
        # Create an array is_peak where is_peak[i] = 1 if nums[i] is a peak (and i not at boundary), else 0.
        is_peak = [0] * n
        def compute_peak(i):
            # Check if index i is a peak given current nums.
            if i == 0 or i == n-1:
                return 0
            return 1 if (nums[i] > nums[i-1] and nums[i] > nums[i+1]) else 0

        for i in range(1, n-1):
            is_peak[i] = compute_peak(i)
        
        # Fenwick Tree implementation.
        size = n
        fenw = [0] * (size+1)
        def fenw_update(i, delta):
            i += 1  # convert to 1-indexed
            while i <= size:
                fenw[i] += delta
                i += i & -i
        
        def fenw_sum(i):
            s = 0
            i += 1
            while i > 0:
                s += fenw[i]
                i -= i & -i
            return s
        
        def fenw_range_sum(l, r):
            return fenw_sum(r) - fenw_sum(l-1)
        
        # Build the Fenwick tree with is_peak array.
        for i in range(size):
            fenw_update(i, is_peak[i])
    
        # A helper function to update a single index's peak value in the fenw tree.
        def update_index(i):
            if i < 1 or i > n-2:  # if it's not possible to be a peak, its value should be 0.
                return
            new_val = compute_peak(i)
            if new_val != is_peak[i]:
                diff = new_val - is_peak[i]
                is_peak[i] = new_val
                fenw_update(i, diff)
    
        res = []
        for q in queries:
            if q[0] == 1:
                # Query type 1: count peaks in subarray nums[l..r]
                l, r = q[1], q[2]
                # Only indices strictly inside the subarray (l+1 to r-1) can be peaks.
                if r - l < 2:
                    res.append(0)
                else:
                    count = fenw_range_sum(l+1, r-1)
                    res.append(count)
            else:
                # Query type 2: update query: set nums[index] = val.
                index, val = q[1], q[2]
                nums[index] = val
                # Update affected indices: index-1, index, index+1 in terms of their peak status.
                update_index(index-1)
                update_index(index)
                update_index(index+1)
    
        return res