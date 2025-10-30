class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # Check if the array has at least 2k elements
        if n < 2 * k:
            return False
        
        # Precompute the increasing status for each possible subarray
        increasing = [False] * (n - k + 1)
        for i in range(n - k + 1):
            is_increasing = True
            for j in range(i, i + k - 1):
                if nums[j] >= nums[j+1]:
                    is_increasing = False
                    break
            increasing[i] = is_increasing
        
        # Check for two adjacent subarrays that are both increasing
        for a in range(len(increasing) - 1):
            b = a + 1
            if increasing[a] and increasing[b]:
                return True
        
        return False