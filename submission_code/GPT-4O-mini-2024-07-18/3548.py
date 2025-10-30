class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from collections import Counter
        
        # Function to check if a number can form a k-palindrome
        def can_form_k_palindrome(counts):
            odd_count = sum(1 for count in counts.values() if count % 2 != 0)
            return odd_count <= 1
        
        # Function to generate all n-digit numbers and count good integers
        def generate_good_integers(n, k):
            good_count = 0
            
            # Generate all n-digit numbers
            for num in range(10**(n-1), 10**n):
                if num % k == 0:
                    str_num = str(num)
                    counts = Counter(str_num)
                    if can_form_k_palindrome(counts):
                        good_count += 1
            
            return good_count
        
        return generate_good_integers(n, k)