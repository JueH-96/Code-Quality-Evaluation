class Fenwick:
    def __init__(self, size):
        self.N = size + 1  # indices from 1 to n
        self.tree = [0] * (self.N + 1)
    
    def update(self, idx, delta):
        while idx < self.N:
            self.tree[idx] += delta
            idx += idx & -idx
    
    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + (nums[i - 1] == max_num)
        
        fenwick = Fenwick(n)
        result = 0
        for right in range(n):
            if right >= k - 1:
                target = prefix[right + 1] - k
                if target >= 0:
                    result += fenwick.query(target)
            fenwick.update(prefix[right + 1], 1)
        return result