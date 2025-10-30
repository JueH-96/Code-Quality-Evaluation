class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        peaks = [0] * n
        
        # Precompute initial peaks
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                peaks[i] = 1
        
        # Initialize Fenwick Tree
        fenwick = FenwickTree(n)
        for i in range(n):
            fenwick.update(i + 1, peaks[i])  # Convert to 1-based index
        
        result = []
        for q in queries:
            if q[0] == 2:
                # Update query: type 2
                index_i = q[1]
                val_i = q[2]
                nums[index_i] = val_i  # Update the value in nums
                
                # Check the positions index_i-1, index_i, index_i+1
                for pos in [index_i - 1, index_i, index_i + 1]:
                    if pos < 0 or pos >= n:
                        continue
                    # Calculate new peak value for pos
                    if pos == 0 or pos == n - 1:
                        new_val = 0
                    else:
                        new_val = 1 if (nums[pos] > nums[pos - 1] and nums[pos] > nums[pos + 1]) else 0
                    delta = new_val - peaks[pos]
                    if delta != 0:
                        fenwick.update(pos + 1, delta)  # Update Fenwick Tree at 1-based index
                        peaks[pos] = new_val
            else:
                # Query for count of peaks in [l, r]
                l = q[1]
                r = q[2]
                length = r - l + 1
                if length < 3:
                    result.append(0)
                else:
                    total = fenwick.query(r) - fenwick.query(l + 1)
                    result.append(total)
        
        return result