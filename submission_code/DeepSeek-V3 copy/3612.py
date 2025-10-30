class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # Precompute the increasing status for each possible subarray
        increasing = [False] * (n - k + 1)
        for i in range(n - k + 1):
            is_increasing = True
            for j in range(1, k):
                if nums[i + j] <= nums[i + j - 1]:
                    is_increasing = False
                    break
            increasing[i] = is_increasing
        # Check for two adjacent increasing subarrays
        for i in range(len(increasing) - 1):
            if increasing[i] and increasing[i + 1]:
                return True
        return False