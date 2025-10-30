class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(n):
            s = str(n)
            return s == s[::-1]
        
        def calculate_cost(nums, target):
            return sum(abs(num - target) for num in nums)
        
        # Generate all palindromes up to 10^9
        palindromes = []
        
        # Single digit palindromes
        for i in range(1, 10):
            palindromes.append(i)
        
        # Multi-digit palindromes
        for length in range(2, 10):  # 2 to 9 digits
            if length % 2 == 0:  # even length
                half_len = length // 2
                start = 10**(half_len - 1)
                end = 10**half_len
                for half in range(start, end):
                    half_str = str(half)
                    palindrome = int(half_str + half_str[::-1])
                    if palindrome < 10**9:
                        palindromes.append(palindrome)
            else:  # odd length
                half_len = length // 2
                start = 10**(half_len - 1) if half_len > 0 else 1
                end = 10**half_len if half_len > 0 else 10
                for half in range(start, end):
                    for middle in range(10):
                        half_str = str(half)
                        if half_len == 0:
                            palindrome = middle
                        else:
                            palindrome = int(half_str + str(middle) + half_str[::-1])
                        if palindrome > 0 and palindrome < 10**9:
                            palindromes.append(palindrome)
        
        # Remove duplicates and sort
        palindromes = sorted(list(set(palindromes)))
        
        # Find minimum cost
        min_cost = float('inf')
        for palindrome in palindromes:
            cost = calculate_cost(nums, palindrome)
            min_cost = min(min_cost, cost)
        
        return min_cost