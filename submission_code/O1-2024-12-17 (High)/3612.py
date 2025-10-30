class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        # We only need to check each starting index i where
        # two consecutive subarrays of length k can fit
        # i.e., i + 2*k - 1 <= n - 1  =>  i <= n - 2*k
        for i in range(n - 2*k + 1):
            # Check if the subarray from i to i+k-1 is strictly increasing
            first_increasing = True
            for j in range(i, i + k - 1):
                if nums[j] >= nums[j + 1]:
                    first_increasing = False
                    break
            
            if not first_increasing:
                continue

            # Check if the subarray from i+k to i+2*k-1 is strictly increasing
            second_increasing = True
            for j in range(i + k, i + 2*k - 1):
                if nums[j] >= nums[j + 1]:
                    second_increasing = False
                    break
            
            if second_increasing:
                return True
        
        return False