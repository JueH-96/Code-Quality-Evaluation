class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        import math
        
        # A helper function that removes square factors from x
        # and returns the remaining square-free part.
        # For example, 12 = 2^2 * 3 --> square-free part = 3.
        # This function finds that by dividing out i^2 if it divides x.
        def square_free_part(x: int) -> int:
            # Remove square factors of small primes ≤ sqrt(x).
            # We'll check all i where i^2 ≤ x and divide out i^2 repeatedly.
            i = 2
            while i * i <= x:
                while x % (i * i) == 0:
                    x //= (i * i)
                i += 1
            return x
        
        # Dictionary: signature -> total sum of elements with that signature
        sum_by_signature = {}
        max_sum = 0
        
        for val in nums:
            signature = square_free_part(val)
            sum_by_signature[signature] = sum_by_signature.get(signature, 0) + val
            max_sum = max(max_sum, sum_by_signature[signature])
        
        return max_sum