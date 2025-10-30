class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        # Sort the array so we can pair the smallest with the largest each time
        nums.sort()
        
        # We'll store the averages in this list
        averages = []
        
        # We pair the i-th smallest with the i-th largest
        for i in range(len(nums)//2):
            avg_val = (nums[i] + nums[-1 - i]) / 2
            averages.append(avg_val)
        
        # Return the minimum value from the collected averages
        return min(averages)