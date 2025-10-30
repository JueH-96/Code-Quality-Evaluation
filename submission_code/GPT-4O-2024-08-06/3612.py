class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        # Check each possible starting index for the first subarray
        for a in range(n - 2 * k + 1):
            # Check if the first subarray is strictly increasing
            first_increasing = True
            for i in range(a, a + k - 1):
                if nums[i] >= nums[i + 1]:
                    first_increasing = False
                    break
            
            if not first_increasing:
                continue
            
            # Check if the second subarray is strictly increasing
            b = a + k
            second_increasing = True
            for i in range(b, b + k - 1):
                if nums[i] >= nums[i + 1]:
                    second_increasing = False
                    break
            
            if second_increasing:
                return True
        
        return False