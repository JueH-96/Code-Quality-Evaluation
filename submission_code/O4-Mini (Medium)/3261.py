from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Compute global OR of nums to know which bits actually appear
        full_or = 0
        for x in nums:
            full_or |= x
        
        # A helper to check if we can achieve OR <= mask using at most k merges
        # Equivalently, we need at least n-k segments whose ANDs have no bits outside mask.
        def can(mask: int) -> bool:
            # forbidden bits are those in full_or but not in mask
            F = full_or & (~mask)
            # Greedy: form as many valid segments as possible (maximizing segment count).
            # A segment is valid iff its bitwise-AND has zero on all bits in F.
            # Equivalently, across the segment there must be at least one element
            # that has a 0 in each forbidden bit position.
            cnt = 0
            i = 0
            while i < n:
                need = F
                j = i
                # extend j until we've zeroed out all forbidden bits
                while j < n and need != 0:
                    # bits that remain needed are those bits where nums[j] has 1
                    need &= nums[j]
                    j += 1
                if need != 0:
                    # reached end but still some forbidden bit never zeroed => fail
                    return False
                # we have a valid segment [i..j-1]
                cnt += 1
                i = j
            # merges used = n - cnt; we need merges_used <= k => cnt >= n - k
            return cnt >= n - k
        
        # Start from all bits set in ans; try to clear bits from high to low
        ans = full_or
        # Only bits up to highest bit in full_or matter; but we'll scan 0..29
        for b in range(29, -1, -1):
            bit = 1 << b
            if ans & bit:
                candidate = ans & ~bit
                if can(candidate):
                    ans = candidate
        return ans

# Example usage:
# sol = Solution()
# print(sol.minOrAfterOperations([3,5,3,2,7], 2))  # expected 3
# print(sol.minOrAfterOperations([7,3,15,14,2,8], 4))  # expected 2
# print(sol.minOrAfterOperations([10,7,10,3,9,14,9,4], 1))  # expected 15