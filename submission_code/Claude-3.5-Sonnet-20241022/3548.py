class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        def is_palindrome(num):
            return str(num) == str(num)[::-1]
            
        def get_digit_freq(num):
            freq = [0] * 10
            for d in str(num):
                freq[int(d)] += 1
            return tuple(freq)
            
        def can_form_palindrome(freq):
            odd_count = sum(1 for f in freq if f % 2)
            return odd_count <= 1
            
        start = 10 ** (n-1) if n > 1 else 0
        end = 10 ** n
        
        good = set()
        
        # Find all k-palindromic numbers in range
        curr = ((start + k - 1) // k) * k
        while curr < end:
            if is_palindrome(curr):
                good.add(get_digit_freq(curr))
            curr += k
            
        # Count numbers that can be rearranged to form k-palindromic numbers
        count = 0
        for num in range(start or 1, end):
            freq = get_digit_freq(num)
            if freq in good:
                count += 1
                
        return count