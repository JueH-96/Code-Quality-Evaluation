from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # ------------ Fenwick Tree helpers ------------
        bit = [0] * (n + 1)            # 1-based internally
        
        def bit_add(i: int, delta: int) -> None:
            """add delta at position i (0-based index)."""
            i += 1                     # shift to 1-based
            while i <= n:
                bit[i] += delta
                i += i & -i
        
        def bit_sum(i: int) -> int:
            """prefix sum [0..i] (0-based)."""
            i += 1
            s = 0
            while i:
                s += bit[i]
                i -= i & -i
            return s
        
        def range_sum(l: int, r: int) -> int:
            """sum of interval [l, r] (0-based, inclusive)."""
            if l > r:
                return 0
            return bit_sum(r) - (bit_sum(l - 1) if l else 0)
        # ------------------------------------------------
        
        # current peak flags
        isPeak = [0] * n
        
        # helper: is position i a peak now?
        def compute_peak(i: int) -> int:
            if 1 <= i <= n - 2 and nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                return 1
            return 0
        
        # build initial peaks and BIT
        for i in range(1, n - 1):
            isPeak[i] = compute_peak(i)
            if isPeak[i]:
                bit_add(i, 1)
        
        ans = []
        
        for q in queries:
            if q[0] == 1:                      # query for count
                l, r = q[1], q[2]
                if r - l < 2:                  # subarray too small to contain peaks
                    ans.append(0)
                else:
                    ans.append(range_sum(l + 1, r - 1))
            
            else:                              # update value
                idx, val = q[1], q[2]
                if nums[idx] == val:           # nothing changes
                    continue
                
                nums[idx] = val               # apply the update first
                
                # only positions idx-1, idx, idx+1 can change their "peak" status
                for j in (idx - 1, idx, idx + 1):
                    if 1 <= j <= n - 2:
                        new_peak = compute_peak(j)
                        if new_peak != isPeak[j]:
                            bit_add(j, new_peak - isPeak[j])
                            isPeak[j] = new_peak
        
        return ans