from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # Fenwick / BIT for prefix sums of peak flags
        class BIT:
            def __init__(self, size: int):
                self.n = size
                self.tree = [0] * (size + 1)
            
            def update(self, i: int, delta: int) -> None:
                # point update: add delta at index i
                while i <= self.n:
                    self.tree[i] += delta
                    i += i & -i
            
            def sum(self, i: int) -> int:
                # prefix sum from 1..i
                s = 0
                while i > 0:
                    s += self.tree[i]
                    i -= i & -i
                return s
        
        bit = BIT(n)
        # is_peak[i] = 1 if nums[i] is a peak (i in [1..n-2]), else 0
        is_peak = [0] * n
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                is_peak[i] = 1
                bit.update(i + 1, 1)
        
        ans = []
        for typ, a, b in queries:
            if typ == 1:
                # query for count of peaks in subarray nums[a..b]
                l, r = a, b
                # peaks can only occur strictly inside (l, r)
                if r - l <= 1:
                    ans.append(0)
                else:
                    # sum of is_peak in indices [l+1 .. r-1]
                    # BIT.sum(x) gives sum of is_peak[0..x-1]
                    total = bit.sum(r) - bit.sum(l + 1)
                    ans.append(total)
            else:
                # update nums[a] = b
                idx, val = a, b
                if nums[idx] == val:
                    # no effective change
                    continue
                nums[idx] = val
                # Only positions idx-1, idx, idx+1 can change peak status
                for j in (idx - 1, idx, idx + 1):
                    if 1 <= j <= n - 2:
                        new_flag = 1 if (nums[j] > nums[j - 1] and nums[j] > nums[j + 1]) else 0
                        if new_flag != is_peak[j]:
                            delta = new_flag - is_peak[j]
                            is_peak[j] = new_flag
                            bit.update(j + 1, delta)
        
        return ans