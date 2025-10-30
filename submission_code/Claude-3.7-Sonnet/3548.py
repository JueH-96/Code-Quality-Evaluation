class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from collections import Counter
        import math
        
        count = 0
        
        # Generate all palindromes of length n
        palindromes = self.generate_palindromes(n)
        
        for palindrome in palindromes:
            if palindrome % k == 0:
                # Count how many n-digit integers have the same digits
                count += self.count_rearrangements(palindrome, n)
        
        return count
    
    def generate_palindromes(self, n: int) -> list:
        palindromes = []
        half_len = (n + 1) // 2
        
        # Generate all possible first halves of the palindrome
        start = 10**(half_len-1)
        end = 10**half_len
        
        for i in range(start, end):
            # Convert to string for easier manipulation
            i_str = str(i)
            
            # Construct the palindrome
            if n % 2 == 0:  # Even length
                palindrome = i_str + i_str[::-1]
            else:  # Odd length
                palindrome = i_str + i_str[:-1][::-1]
            
            palindromes.append(int(palindrome))
        
        return palindromes
    
    def count_rearrangements(self, palindrome: int, n: int) -> int:
        # Count frequencies of each digit
        palindrome_str = str(palindrome)
        multiset = Counter(palindrome_str)
        
        # Calculate total permutations of these digits
        total_permutations = math.factorial(n)
        for digit, freq in multiset.items():
            total_permutations //= math.factorial(freq)
        
        # Subtract permutations with leading zero
        permutations_with_leading_zero = 0
        if '0' in multiset:
            zero_freq = multiset['0']
            permutations_with_leading_zero = math.factorial(n-1)
            permutations_with_leading_zero //= math.factorial(zero_freq-1)
            for digit, freq in multiset.items():
                if digit != '0':
                    permutations_with_leading_zero //= math.factorial(freq)
        
        return total_permutations - permutations_with_leading_zero