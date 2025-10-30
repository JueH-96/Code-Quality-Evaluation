class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        for a in range(n - 2 * k + 1):
            b = a + k
            
            # Check if subarray starting at index a is strictly increasing
            is_subarray_a_increasing = True
            for i in range(a, a + k - 1):
                if nums[i] >= nums[i + 1]:
                    is_subarray_a_increasing = False
                    break
            
            # Check if subarray starting at index b is strictly increasing
            is_subarray_b_increasing = True
            for i in range(b, b + k - 1):
                if nums[i] >= nums[i + 1]:
                    is_subarray_b_increasing = False
                    break
            
            if is_subarray_a_increasing and is_subarray_b_increasing:
                return True
        
        return False