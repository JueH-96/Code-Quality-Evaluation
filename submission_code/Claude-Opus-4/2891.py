class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # Sort the array
        nums.sort()
        
        # Use two pointers to find the maximum window where 
        # all elements can be made equal
        left = 0
        max_beauty = 0
        
        for right in range(len(nums)):
            # Check if nums[left] and nums[right] can be made equal
            # They can be made equal if their ranges overlap
            # i.e., if nums[right] - nums[left] <= 2 * k
            while nums[right] - nums[left] > 2 * k:
                left += 1
            
            max_beauty = max(max_beauty, right - left + 1)
        
        return max_beauty