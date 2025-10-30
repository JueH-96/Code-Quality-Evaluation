class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def generate_palindromes(max_val):
            palindromes = []
            
            # 1-digit palindromes
            for i in range(1, 10):
                palindromes.append(i)
            
            # Multi-digit palindromes
            for length in range(2, 10):  # up to 9 digits
                if length % 2 == 0:  # even length
                    half_length = length // 2
                    start = 10**(half_length-1)
                    end = 10**half_length
                    for i in range(start, end):
                        s = str(i)
                        palindrome = int(s + s[::-1])
                        if palindrome < max_val:
                            palindromes.append(palindrome)
                else:  # odd length
                    half_length = length // 2
                    start = 10**(half_length-1)
                    end = 10**half_length
                    for i in range(start, end):
                        s = str(i)
                        for mid in range(10):
                            palindrome = int(s + str(mid) + s[::-1])
                            if palindrome < max_val:
                                palindromes.append(palindrome)
            
            return palindromes
        
        palindromes = generate_palindromes(10**9)
        min_cost = float('inf')
        
        for p in palindromes:
            cost = sum(abs(num - p) for num in nums)
            min_cost = min(min_cost, cost)
        
        return min_cost