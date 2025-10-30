class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def digit_sum(n: str) -> int:
            return sum(int(digit) for digit in n)
        
        def count_good_numbers(n: str) -> int:
            if not n:
                return 0
            count = 0
            for i in range(1, int(n[0])):
                count += count_good_numbers(str(i) + '0' * (len(n) - 1))
            count += count_good_numbers(n[1:])
            if min_sum <= digit_sum(n) <= max_sum:
                count += 1
            return count % MOD
        
        count_num2 = count_good_numbers(num2)
        count_num1_minus_1 = count_good_numbers(str(int(num1) - 1))
        
        return (count_num2 - count_num1_minus_1) % MOD