class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def digit_sum(x):
            return sum(int(d) for d in str(x))
        
        start = int(num1)
        end = int(num2)
        count_good = 0
        
        for x in range(start, end + 1):
            d_sum = digit_sum(x)
            if min_sum <= d_sum <= max_sum:
                count_good += 1
        
        return count_good % MOD