class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        def is_increasing(subarray):
            return all(subarray[i] < subarray[i+1] for i in range(len(subarray)-1))
        
        n = len(nums)
        max_a = n - 2 * k
        for a in range(max_a + 1):
            first_subarray = nums[a:a+k]
            second_subarray = nums[a+k:a+2*k]
            if is_increasing(first_subarray) and is_increasing(second_subarray):
                return True
        return False