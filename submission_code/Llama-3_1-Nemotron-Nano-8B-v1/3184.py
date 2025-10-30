import bisect
from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        x_list = [nums[i] - i for i in range(len(nums))]
        sorted_unique_x = sorted(set(x_list))
        m = len(sorted_unique_x)
        
        def get_rank(x):
            return bisect.bisect_left(sorted_unique_x, x) + 1  # 1-based rank
        
        class FenwickTree:
            def __init__(self, size):
                self.n = size
                self.tree = [0] * (self.n + 1)  # 1-based indexing
            
            def update(self, idx, value):
                while idx <= self.n:
                    if self.tree[idx] < value:
                        self.tree[idx] = value
                    else:
                        break
                    idx += idx & -idx
            
            def query(self, idx):
                res = 0
                while idx > 0:
                    if res < self.tree[idx]:
                        res = self.tree[idx]
                    idx -= idx & -idx
                return res
        
        ft = FenwickTree(m)
        
        for i in range(len(nums)):
            x = x_list[i]
            rank = get_rank(x)
            prev_max = ft.query(rank)
            current_sum = max(prev_max + nums[i], nums[i])
            ft.update(rank, current_sum)
        
        return ft.query(m)