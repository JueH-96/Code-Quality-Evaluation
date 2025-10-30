from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # Fenwick / BIT for range sum over isPeak array
        class Fenwick:
            def __init__(self, n: int):
                self.n = n
                self.fw = [0] * n
            
            # add delta at index i
            def update(self, i: int, delta: int):
                while i < self.n:
                    self.fw[i] += delta
                    # move to next
                    i |= i + 1
            
            # sum of [0..i]
            def query(self, i: int) -> int:
                s = 0
                while i >= 0:
                    s += self.fw[i]
                    # move to parent
                    i = (i & (i + 1)) - 1
                return s
            
            # sum of [l..r]
            def range_query(self, l: int, r: int) -> int:
                if l > r:
                    return 0
                if l == 0:
                    return self.query(r)
                return self.query(r) - self.query(l - 1)
        
        # helper to check if index i is a peak in current nums
        def is_peak(i: int) -> int:
            # cannot be first or last
            if i <= 0 or i >= n - 1:
                return 0
            return 1 if (nums[i] > nums[i-1] and nums[i] > nums[i+1]) else 0
        
        # build initial isPeak array and Fenwick tree
        bit = Fenwick(n)
        # isPeak array to remember old state
        isPeak = [0] * n
        for i in range(1, n-1):
            p = is_peak(i)
            isPeak[i] = p
            if p:
                bit.update(i, p)
        
        ans = []
        
        for q in queries:
            t = q[0]
            if t == 1:
                l, r = q[1], q[2]
                # subarray of length < 3 has no peaks
                if r - l < 2:
                    ans.append(0)
                else:
                    # count peaks in (l, r), i.e. indices l+1..r-1
                    cnt = bit.range_query(l+1, r-1)
                    ans.append(cnt)
            else:
                # update nums[index] = val
                idx, val = q[1], q[2]
                # apply update
                nums[idx] = val
                # re-check peak status for idx-1, idx, idx+1
                for j in (idx-1, idx, idx+1):
                    if 1 <= j <= n-2:
                        newp = is_peak(j)
                        if newp != isPeak[j]:
                            # update bit by the difference
                            diff = newp - isPeak[j]
                            bit.update(j, diff)
                            isPeak[j] = newp
        
        return ans