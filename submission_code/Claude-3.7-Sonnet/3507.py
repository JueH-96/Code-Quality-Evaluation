class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_prime(n):
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        # Count special numbers (prime squares) within the range [l, r]
        special_count = 0
        for i in range(2, int(r ** 0.5) + 1):
            if is_prime(i) and l <= i*i <= r:
                special_count += 1
        
        # Total range size minus special numbers
        return r - l + 1 - special_count