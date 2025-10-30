class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        
        while nums:
            # Get the largest numbers from each row
            max_values = [max(row) for row in nums]
            # Find the highest number among the max values
            highest = max(max_values)
            # Add that number to the score
            score += highest
            
            # Remove the highest number from each row
            for i in range(len(nums)):
                nums[i].remove(highest)
            
            # Remove empty rows
            nums = [row for row in nums if row]
        
        return score