from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # Count how many times each bit appears across all nums
        # bits up to 30 (since nums[i] <= 1e9 < 2^30)
        MAXB = 30
        cnt = [0] * (MAXB + 1)
        for num in nums:
            # count bits in num
            for b in range(MAXB + 1):
                if num >> b & 1:
                    cnt[b] += 1
        
        # We will build the k largest final numbers in arr (sorted descending).
        # Initially all zero.
        arr = [0] * k
        
        # Process bits from high to low. For bit b, we can assign min(cnt[b], k)
        # ones to our k target numbers. To maximize sum of squares, give these
        # bits to the currently largest arr entries (prefix in descending array).
        for b in range(MAXB, -1, -1):
            t = cnt[b]
            if t > k:
                t = k
            if t <= 0:
                continue
            v = 1 << b
            # arr is kept sorted descending. We add v to the first t entries.
            for i in range(t):
                arr[i] += v
            # after this, arr[0..t-1] increased; since they were the largest,
            # they stay >= arr[t], so arr remains sorted descending.
        
        # Compute sum of squares of these k numbers, modulo MOD
        ans = 0
        for x in arr:
            ans = (ans + x * x) % MOD
        return ans