class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def digit_sum(x):
            return sum(int(d) for d in str(x))
        
        def count_good_integers(low, high):
            count = 0
            for x in range(low, high + 1):
                d_sum = digit_sum(x)
                if min_sum <= d_sum <= max_sum:
                    count += 1
            return count
        
        low = int(num1)
        high = int(num2)
        
        return count_good_integers(low, high) % MOD