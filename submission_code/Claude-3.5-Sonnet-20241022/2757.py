class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 1000000007
        
        def digit_sum(n: str) -> int:
            return sum(int(d) for d in n)
        
        def count_up_to(num: str, min_s: int, max_s: int, memo={}) -> int:
            if not num:
                return 1 if min_s <= 0 and 0 <= max_s else 0
            
            key = (num, min_s, max_s)
            if key in memo:
                return memo[key]
            
            result = 0
            curr_sum = digit_sum(num)
            first_digit = int(num[0])
            
            # Try placing smaller digits
            for d in range(first_digit):
                remaining_min = min_s - d
                remaining_max = max_s - d
                result = (result + count_up_to('9' * (len(num)-1), 
                         remaining_min, remaining_max)) % MOD
            
            # Try placing the same digit
            remaining_min = min_s - first_digit
            remaining_max = max_s - first_digit
            result = (result + count_up_to(num[1:], 
                     remaining_min, remaining_max)) % MOD
            
            memo[key] = result
            return result
        
        def count_valid(num: str) -> int:
            result = 0
            for i in range(len(num)):
                prefix = num[:i]
                if prefix and prefix[0] == '0':
                    continue
                curr_sum = digit_sum(prefix) if prefix else 0
                for d in range(10):
                    if i == 0 and d == 0 and len(num) > 1:
                        continue
                    new_sum = curr_sum + d
                    if min_sum <= new_sum <= max_sum:
                        result = (result + 1) % MOD
            return result
        
        # Handle single digit numbers
        if len(num1) == 1 and len(num2) == 1:
            count = 0
            for i in range(int(num1), int(num2) + 1):
                if min_sum <= i <= max_sum:
                    count += 1
            return count
        
        # Count numbers with different lengths
        result = 0
        for length in range(len(num1), len(num2) + 1):
            if length == len(num1):
                start = num1
            else:
                start = '1' + '0' * (length - 1)
                
            if length == len(num2):
                end = num2
            else:
                end = '9' * length
                
            # Count from start to end
            result = (result + count_up_to(end, min_sum, max_sum) - 
                     count_up_to(str(int(start) - 1), min_sum, max_sum)) % MOD
            
        return result