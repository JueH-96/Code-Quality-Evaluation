class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the minimum and maximum elements in the array
        min_val, max_val = min(nums), max(nums)
        
        # Check if the array is already equalindromic
        if all(num == min_val for num in nums):
            return 0
        
        # Find the closest palindromic number to the average of the array
        avg = sum(nums) / n
        closest_palindrome = self.find_closest_palindrome(avg)
        
        # Calculate the minimum cost to make the array equalindromic
        total_cost = 0
        for num in nums:
            total_cost += abs(num - closest_palindrome)
        
        return total_cost
    
    def is_palindrome(self, num: int) -> bool:
        """
        Helper function to check if a number is palindromic.
        """
        return str(num) == str(num)[::-1]
    
    def find_closest_palindrome(self, target: float) -> int:
        """
        Helper function to find the closest palindromic number to a given target.
        """
        closest_palindrome = round(target)
        while not self.is_palindrome(closest_palindrome):
            if closest_palindrome < target:
                closest_palindrome += 1
            else:
                closest_palindrome -= 1
        return closest_palindrome