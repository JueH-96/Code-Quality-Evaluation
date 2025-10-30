class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        nums.sort()  # Sort the array in ascending order
        
        while nums:
            min_element = nums.pop(0)  # Remove the smallest element
            max_element = nums.pop()   # Remove the largest element
            average = (min_element + max_element) / 2
            averages.append(average)
        
        return min(averages)