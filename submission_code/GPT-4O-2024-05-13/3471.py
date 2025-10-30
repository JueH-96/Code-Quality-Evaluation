class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        nums.sort()
        
        while nums:
            minElement = nums.pop(0)
            maxElement = nums.pop(-1)
            averages.append((minElement + maxElement) / 2)
        
        return min(averages)