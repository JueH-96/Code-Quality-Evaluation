class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        nums = nums.copy()
        
        while nums:
            min_element = min(nums)
            nums.remove(min_element)
            max_element = max(nums) 
            nums.remove(max_element)
            
            avg = (min_element + max_element) / 2
            averages.append(avg)
            
        return min(averages)