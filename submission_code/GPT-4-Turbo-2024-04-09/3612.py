class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        def is_strictly_increasing(subarray):
            return all(subarray[i] < subarray[i+1] for i in range(len(subarray) - 1))
        
        for i in range(len(nums) - 2 * k + 1):
            first_subarray = nums[i:i+k]
            second_subarray = nums[i+k:i+2*k]
            if is_strictly_increasing(first_subarray) and is_strictly_increasing(second_subarray):
                return True
        return False