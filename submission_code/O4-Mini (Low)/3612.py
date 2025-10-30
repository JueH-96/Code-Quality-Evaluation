from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # inc_len[i] will store the length of the strictly increasing run starting at i
        inc_len = [1] * n
        # Build inc_len from the end
        for i in range(n - 2, -1, -1):
            if nums[i] < nums[i + 1]:
                inc_len[i] = inc_len[i + 1] + 1
            else:
                inc_len[i] = 1
        
        # Check for any a such that both runs at a and a+k are at least k long
        # and the subarrays are adjacent => second starts at a + k
        for a in range(0, n - 2 * k + 1):
            if inc_len[a] >= k and inc_len[a + k] >= k:
                return True
        
        return False