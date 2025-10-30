class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_element = max(nums)
        count = 0
        
        for i in range(len(nums)):
            max_count = 0
            for j in range(i, len(nums)):
                if nums[j] == max_element:
                    max_count += 1
                if max_count >= k:
                    # Once we have k occurrences of max_element,
                    # any subarray from i to j′ (where j′ ≥ j) will qualify
                    count += len(nums) - j
                    break
        
        return count