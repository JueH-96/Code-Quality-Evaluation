from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        # Total number of subarrays
        total = n * (n + 1) // 2
        # 0-based index of the median in the sorted list of all distinct-counts
        K = (total - 1) // 2
        
        # We will binary-search the answer d in [1..n], looking
        # for the smallest d such that the number of subarrays
        # with at most d distinct elements is > K.
        
        # We'll need a routine count_at_most(d) that returns
        # the number of subarrays of nums having <= d distinct elems.
        # We implement it with the standard two-pointer + freq table,
        # optimized by a timestamp trick so we can reuse one freq array
        # without clearing it fully each call.
        
        maxv = max(nums)
        freq = [0] * (maxv + 1)
        last = [0] * (maxv + 1)
        # timestamp
        ts = 0
        
        def count_at_most(d: int) -> int:
            nonlocal ts
            ts += 1
            cur_ts = ts
            distinct = 0
            left = 0
            cnt = 0
            for right, x in enumerate(nums):
                # add nums[right]
                if last[x] != cur_ts:
                    # first time in this run
                    last[x] = cur_ts
                    freq[x] = 1
                    distinct += 1
                else:
                    freq[x] += 1
                # shrink from left if too many distinct
                while distinct > d:
                    y = nums[left]
                    freq[y] -= 1
                    if freq[y] == 0:
                        distinct -= 1
                    left += 1
                # now [left..right] has <= d distinct
                cnt += (right - left + 1)
            return cnt
        
        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            if count_at_most(mid) > K:
                hi = mid
            else:
                lo = mid + 1
        return lo