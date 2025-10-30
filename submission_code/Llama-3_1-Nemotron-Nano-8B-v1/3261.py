from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        res = 0
        n = len(nums)
        for bit in reversed(range(31)):  # from 30 downto 0
            res |= (nums[0] >> bit) & 1
            mask = 1 << bit
            # Check if there's at least one element without this bit set
            has_non_set = any((x & mask) == 0 for x in nums)
            if not has_non_set:
                continue
            # Compute the number of elements with this bit set
            cnt = sum(1 for x in nums if (x & mask) != 0)
            if cnt <= k:
                res ^= mask  # clear the bit
                k -= cnt
            if k <= 0:
                break
        return res