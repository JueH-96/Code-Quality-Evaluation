class Solution:
    def isBalanced(self, num: str) -> bool:
        # Get digits at even indices
        even_indices = num[::2]
        # Get digits at odd indices
        odd_indices = num[1::2]
        
        # Calculate sum of digits at even indices
        sum_even = sum(int(char) for char in even_indices)
        # Calculate sum of digits at odd indices
        sum_odd = sum(int(char) for char in odd_indices)
        
        # Return True if sums are equal, else False
        return sum_even == sum_odd