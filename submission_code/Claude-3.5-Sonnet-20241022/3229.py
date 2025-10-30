class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(n):
            return str(n) == str(n)[::-1]
            
        def get_cost(target):
            return sum(abs(x - target) for x in nums)
            
        # Find min and max values in nums
        min_val = min(nums)
        max_val = max(nums)
        
        # Generate palindromes around the average
        avg = sum(nums) // len(nums)
        palindromes = []
        
        # Check palindromes below average
        curr = avg
        while curr >= 1:
            if is_palindrome(curr):
                palindromes.append(curr)
            if curr == 1:
                break
            curr -= 1
            
        # Check palindromes above average    
        curr = avg + 1
        while curr <= max(max_val + 1, 10**5):
            if is_palindrome(curr):
                palindromes.append(curr)
            if len(str(curr)) > len(str(max_val)):
                break
            curr += 1
            
        # Find minimum cost among all palindromes
        min_cost = float('inf')
        for p in palindromes:
            cost = get_cost(p)
            min_cost = min(min_cost, cost)
            
        return min_cost