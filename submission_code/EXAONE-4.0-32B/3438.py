from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        self.n = n
        self.nums = nums
        self.peaks = [0] * n
        
        for i in range(1, n-1):
            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                self.peaks[i] = 1
        
        class Fenw:
            def __init__(self, size):
                self.n = size
                self.tree = [0] * (self.n + 1)
            
            def update(self, index, delta):
                i = index + 1
                while i <= self.n:
                    self.tree[i] += delta
                    i += i & -i
            
            def query(self, index):
                if index < 0:
                    return 0
                i = index + 1
                s = 0
                while i:
                    s += self.tree[i]
                    i -= i & -i
                return s
        
        self.fenw = Fenw(n)
        for i in range(n):
            if self.peaks[i]:
                self.fenw.update(i, 1)
        
        ans = []
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                left_idx = l + 1
                right_idx = r - 1
                if left_idx > right_idx:
                    ans.append(0)
                else:
                    res = self.fenw.query(right_idx) - self.fenw.query(left_idx - 1)
                    ans.append(res)
            else:
                idx, val = q[1], q[2]
                if self.nums[idx] == val:
                    continue
                old_val = self.nums[idx]
                self.nums[idx] = val
                indices_to_check = set([idx-1, idx, idx+1])
                for i in indices_to_check:
                    if i < 1 or i >= n-1:
                        continue
                    new_cond = (self.nums[i] > self.nums[i-1]) and (self.nums[i] > self.nums[i+1])
                    new_peak = 1 if new_cond else 0
                    if self.peaks[i] != new_peak:
                        delta = new_peak - self.peaks[i]
                        self.fenw.update(i, delta)
                        self.peaks[i] = new_peak
        return ans