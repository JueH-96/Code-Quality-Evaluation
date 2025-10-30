class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def digit_sum(x):
            return sum(int(d) for d in str(x))
        
        def count_good_numbers_up_to(n):
            if n < 0:
                return 0
            count = 0
            for i in range(int(num1), n + 1):
                if min_sum <= digit_sum(i) <= max_sum:
                    count += 1
            return count
        
        num1_int = int(num1)
        num2_int = int(num2)
        
        result = count_good_numbers_up_to(num2_int) - count_good_numbers_up_to(num1_int - 1)
        return result % MOD