from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        is_peak = [False] * n
        
        # Initialize the is_peak array
        for i in range(n):
            if i == 0 or i == n - 1:
                is_peak[i] = False
            else:
                is_peak[i] = nums[i] > nums[i - 1] and nums[i] > nums[i + 1]
        
        # Fenwick Tree implementation
        class FenwickTree:
            def __init__(self, size):
                self.n = size
                self.tree = [0] * (self.n + 1)  # 1-based indexing
            
            def update_point(self, idx, delta):
                while idx <= self.n:
                    self.tree[idx] += delta
                    idx += idx & -idx
            
            def query_prefix(self, idx):
                res = 0
                while idx > 0:
                    res += self.tree[idx]
                    idx -= idx & -idx
                return res
        
        fenwick = FenwickTree(n)
        for i in range(n):
            val = 1 if is_peak[i] else 0
            fenwick.update_point(i + 1, val)
        
        answer = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                a = l + 1
                b = r - 1
                if a > b:
                    answer.append(0)
                else:
                    res = fenwick.query_prefix(b + 1) - fenwick.query_prefix(a)
                    answer.append(res)
            else:
                index = q[1]
                val = q[2]
                if nums[index] == val:
                    continue
                nums[index] = val
                # Check neighbors
                for j in [index - 1, index, index + 1]:
                    if 0 <= j < n:
                        if j == 0 or j == n - 1:
                            new_peak = False
                        else:
                            new_peak = nums[j] > nums[j - 1] and nums[j] > nums[j + 1]
                        if new_peak != is_peak[j]:
                            delta = 1 if new_peak else -1
                            fenwick.update_point(j + 1, delta)
                            is_peak[j] = new_peak
        return answer