from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        # We want to maximize K such that we can form K pairs (i, j) with nums[i] < nums[j],
        # using each element at most once. A valid strategy on a sorted array is to pair
        # the K smallest with the K largest:
        #   (nums[0], nums[n-K]), (nums[1], nums[n-K+1]), ..., (nums[K-1], nums[n-1])
        # We binary search on K in [0 .. n//2].
        
        lo, hi = 0, n // 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            # Check if we can make mid valid pairs
            ok = True
            # for each 0 <= i < mid, check nums[i] < nums[n-mid+i]
            for i in range(mid):
                if nums[i] >= nums[n - mid + i]:
                    ok = False
                    break
            if ok:
                lo = mid
            else:
                hi = mid - 1
        
        # lo is the maximum number of pairs we can remove
        return n - 2 * lo