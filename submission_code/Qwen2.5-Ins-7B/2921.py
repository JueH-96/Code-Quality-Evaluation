class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def is_stepping_number(num):
            num_str = str(num)
            for i in range(1, len(num_str)):
                if abs(int(num_str[i]) - int(num_str[i-1])) != 1:
                    return False
            return True
        
        def count_numbers_with_length(n, leading_digit):
            if n == 1:
                return 1
            count = 0
            for next_digit in range(leading_digit - 1, leading_digit + 2):
                if 0 <= next_digit <= 9:
                    count += count_numbers_with_length(n - 1, next_digit)
            return count
        
        def count_stepping_numbers_up_to(n):
            n_str = str(n)
            count = 0
            for length in range(1, len(n_str) + 1):
                for start_digit in range(1, 10):
                    if length == len(n_str) and int(n_str[0]) != start_digit:
                        continue
                    count += count_numbers_with_length(length, start_digit)
                    if count_numbers_with_length(length, start_digit) > 0 and is_stepping_number(int(n_str[:length])):
                        count -= 1
            return count
        
        low_num = int(low)
        high_num = int(high)
        count = count_stepping_numbers_up_to(high_num) - count_stepping_numbers_up_to(low_num - 1)
        return count % MOD