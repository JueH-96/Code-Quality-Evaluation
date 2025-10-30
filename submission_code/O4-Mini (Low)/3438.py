from typing import List

class Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.fw = [0] * (n + 1)
        
    def update(self, i: int, delta: int) -> None:
        # point update at index i (0-based)
        i += 1
        while i <= self.n:
            self.fw[i] += delta
            i += i & -i
        
    def query(self, i: int) -> int:
        # prefix sum [0..i]
        i += 1
        s = 0
        while i > 0:
            s += self.fw[i]
            i -= i & -i
        return s
    
    def range_sum(self, l: int, r: int) -> int:
        if l > r:
            return 0
        return self.query(r) - (self.query(l - 1) if l > 0 else 0)

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # b[i] = 1 if nums[i] is a peak (i in [1..n-2]), else 0
        b = [0] * n
        def is_peak(i: int) -> int:
            if 1 <= i <= n - 2 and nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                return 1
            return 0
        
        # build initial b
        for i in range(1, n-1):
            b[i] = is_peak(i)
        
        # fenwick tree over b
        fw = Fenwick(n)
        for i in range(n):
            if b[i]:
                fw.update(i, b[i])
        
        ans = []
        for typ, x, y in queries:
            if typ == 1:
                l, r = x, y
                # peaks in (l, r), i.e. indices l+1..r-1
                left = l + 1
                right = r - 1
                # also ensure within [1..n-2]
                left = max(left, 1)
                right = min(right, n-2)
                ans.append(fw.range_sum(left, right))
            else:
                # update nums[x] = y
                idx, val = x, y
                # change nums[idx]
                nums[idx] = val
                # positions that could change: idx-1, idx, idx+1
                for i in (idx-1, idx, idx+1):
                    if 1 <= i <= n-2:
                        newb = is_peak(i)
                        if newb != b[i]:
                            fw.update(i, newb - b[i])
                            b[i] = newb
        return ans