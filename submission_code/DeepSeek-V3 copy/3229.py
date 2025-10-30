class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Generate all palindromic numbers less than 10^9
        palindromes = set()
        
        # Generate single-digit palindromes
        for i in range(1, 10):
            palindromes.add(i)
        
        # Generate two-digit palindromes
        for i in range(1, 10):
            palindromes.add(int(str(i) + str(i)))
        
        # Generate palindromes with more digits
        for length in range(3, 10):
            half_length = (length + 1) // 2
            for i in range(10 ** (half_length - 1), 10 ** half_length):
                s = str(i)
                if length % 2 == 0:
                    palindrome = int(s + s[::-1])
                else:
                    palindrome = int(s + s[:-1][::-1])
                if palindrome < 10**9:
                    palindromes.add(palindrome)
        
        # Convert to sorted list
        palindromes = sorted(palindromes)
        
        # Find the palindrome that minimizes the total cost
        min_cost = float('inf')
        for p in palindromes:
            total_cost = 0
            for num in nums:
                total_cost += abs(num - p)
            if total_cost < min_cost:
                min_cost = total_cost
            if min_cost == 0:
                break
        
        return min_cost