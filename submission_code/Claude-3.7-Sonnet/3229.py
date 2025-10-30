class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Function to check if a number is palindromic
        def is_palindrome(n):
            return str(n) == str(n)[::-1]
        
        # Function to find palindromic numbers around a target value
        def find_nearby_palindromes(target, count=100):
            palindromes = []
            offset = 0
            while len(palindromes) < count:
                # Check values above target
                if target + offset > 0 and is_palindrome(target + offset):
                    palindromes.append(target + offset)
                
                # Check values below target
                if offset > 0 and target - offset > 0 and is_palindrome(target - offset):
                    palindromes.append(target - offset)
                
                offset += 1
            
            return palindromes
        
        # Find the median of the array
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        
        if n % 2 == 1:
            median = nums_sorted[n // 2]
        else:
            median = (nums_sorted[n // 2 - 1] + nums_sorted[n // 2]) // 2
        
        # Find palindromic numbers near the median
        palindromes = find_nearby_palindromes(median)
        
        # Calculate the cost for each palindromic number
        min_cost = float('inf')
        for p in palindromes:
            cost = sum(abs(num - p) for num in nums)
            min_cost = min(min_cost, cost)
        
        return min_cost