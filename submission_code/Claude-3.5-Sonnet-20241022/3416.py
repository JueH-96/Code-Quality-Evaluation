class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        total = 0
        n = len(nums)
        
        # Convert all numbers to strings for easier digit comparison
        nums = [str(num) for num in nums]
        
        # Compare each pair of numbers
        for i in range(n):
            for j in range(i+1, n):
                # Count different digits at same positions
                diff = sum(1 for x, y in zip(nums[i], nums[j]) if x != y)
                total += diff
                
        return total