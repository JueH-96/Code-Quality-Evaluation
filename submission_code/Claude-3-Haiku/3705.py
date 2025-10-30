from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        # Create a dictionary to store the count of each integer in each subarray of size k
        count = defaultdict(lambda: [0] * len(nums))
        
        # Iterate through the array and update the count dictionary
        for i in range(len(nums)):
            for j in range(max(0, i-k+1), i+1):
                count[nums[i]][j] += 1
        
        # Find the largest integer that appears in exactly one subarray of size k
        max_integer = -1
        for num in range(50, -1, -1):
            if sum(count[num]) == 1:
                max_integer = num
                break
        
        return max_integer