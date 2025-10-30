class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n+1)  # Difference array to track how many times we've subtracted up to an index
        current_sub = 0     # Current net subtractions applied to the current index

        for i in range(n):
            current_sub += diff[i]     # Update current_sub based on what ends at i
            val = nums[i] - current_sub
            if val < 0:
                return False           # Can't fix if it's already below 0
            if val > 0:
                if i + k > n:         # Not enough room to apply k-length subarray
                    return False
                current_sub += val    # Apply val extra subtractions starting at i
                diff[i + k] -= val    # Those subtractions will end right after i+k-1

        return True