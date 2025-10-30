class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10 ** 9 + 7
        
        def digit_sum(x):
            return sum(int(d) for d in str(x))
        
        count = 0
        for x in range(int(num1), int(num2) + 1):
            if min_sum <= digit_sum(x) <= max_sum:
                count += 1
        
        return count % MOD