import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [float('-inf')] * (self.n + 1)  # 1-based indexing
    
    def update(self, idx, value):
        while idx <= self.n:
            if self.tree[idx] < value:
                self.tree[idx] = value
            else:
                break  # No need to proceed further
            idx += idx & -idx
    
    def query(self, idx):
        res = float('-inf')
        while idx > 0:
            if self.tree[idx] > res:
                res = self.tree[idx]
            idx -= idx & -idx
        return res

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        a = [nums[i] - i for i in range(n)]
        
        # Coordinate compression
        sorted_a = sorted(a)
        unique_a = []
        prev = None
        for num in sorted_a:
            if num != prev:
                unique_a.append(num)
                prev = num
        m = len(unique_a)
        
        fenwick = FenwickTree(m)
        max_sum = float('-inf')
        
        for i in range(n):
            ai = a[i]
            # Find rank in unique_a
            rank = bisect.bisect_left(unique_a, ai)
            fenwick_idx = rank + 1  # convert to 1-based index
            
            current_max = fenwick.query(fenwick_idx)
            current_dp = nums[i] + max(current_max, 0)
            
            max_sum = max(max_sum, current_dp)
            
            fenwick.update(fenwick_idx, current_dp)
        
        return max_sum