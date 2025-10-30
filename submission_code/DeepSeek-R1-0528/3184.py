class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [-10**18] * (self.n + 1)
    
    def update(self, idx, val):
        while idx <= self.n:
            if val > self.tree[idx]:
                self.tree[idx] = val
            idx += idx & -idx
            
    def query(self, idx):
        res = -10**18
        while idx:
            if self.tree[idx] > res:
                res = self.tree[idx]
            idx -= idx & -idx
        return res

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        a = [nums[i] - i for i in range(n)]
        unique_vals = sorted(set(a))
        comp = {val: idx + 1 for idx, val in enumerate(unique_vals)}
        size = len(unique_vals)
        fenw = Fenw(size)
        ans = -10**18
        
        for i in range(n):
            r = comp[a[i]]
            max_val = fenw.query(r)
            current = nums[i] + max(0, max_val)
            if current > ans:
                ans = current
            fenw.update(r, current)
        
        return ans