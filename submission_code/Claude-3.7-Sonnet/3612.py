class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        # Iterate through all possible starting positions for the first subarray
        for i in range(n - 2 * k + 1):
            # Check if the first subarray is strictly increasing
            first_valid = True
            for j in range(i, i + k - 1):
                if nums[j] >= nums[j + 1]:
                    first_valid = False
                    break
            
            if not first_valid:
                continue
            
            # Check if the second subarray is strictly increasing
            second_valid = True
            for j in range(i + k, i + 2 * k - 1):
                if nums[j] >= nums[j + 1]:
                    second_valid = False
                    break
            
            # If both subarrays are strictly increasing, return True
            if second_valid:
                return True
        
        return False