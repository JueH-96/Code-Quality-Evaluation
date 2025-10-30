class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(x):
            return str(x) == str(x)[::-1]
        
        # Generate all palindromic numbers less than 10^9
        palindromes = []
        for i in range(1, 10**9):
            if is_palindrome(i):
                palindromes.append(i)
            if i > 10**9:
                break
        
        # Find the minimum and maximum values in nums to limit the search space
        min_num = min(nums)
        max_num = max(nums)
        
        # Filter palindromes to only those within the range of the numbers in nums
        palindromes = [p for p in palindromes if min_num <= p <= max_num]
        
        # If no palindromes in range, return impossible (though problem constraints imply there should be)
        if not palindromes:
            return float('inf')
        
        # Calculate the cost for each palindrome and find the minimum cost
        min_cost = float('inf')
        for p in palindromes:
            cost = sum(abs(num - p) for num in nums)
            if cost < min_cost:
                min_cost = cost
        
        return min_cost