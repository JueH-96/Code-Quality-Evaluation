class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = n
        target_endings = ["00", "25", "50", "75"]
        for ending in target_endings:
            digit2 = ending[1]
            digit1 = ending[0]
            last_index_digit2 = -1
            for i in range(n - 1, -1, -1):
                if num[i] == digit2:
                    last_index_digit2 = i
                    break
            if last_index_digit2 == -1:
                continue
            last_index_digit1 = -1
            for i in range(last_index_digit2 - 1, -1, -1):
                if num[i] == digit1:
                    last_index_digit1 = i
                    break
            if last_index_digit1 == -1:
                continue
            current_ops = n - (last_index_digit1 + 2)
            min_ops = min(min_ops, current_ops)
            
        # Special case: getting "0"
        zeros_count = 0
        for digit in num:
            if digit == '0':
                zeros_count += 1
        
        non_zero_digits_count = 0
        for digit in num:
            if digit != '0':
                non_zero_digits_count += 1
        
        min_ops = min(min_ops, non_zero_digits_count)
        
        # If the original number is already divisible by 25, 0 operations.
        if n >= 2:
            last_two_digits_str = ""
            found_last_digit = False
            found_second_last_digit = False
            for i in range(n - 1, -1, -1):
                if not found_last_digit:
                    last_two_digits_str = num[i] + last_two_digits_str
                    found_last_digit = True
                elif not found_second_last_digit:
                    last_two_digits_str = num[i] + last_two_digits_str
                    found_second_last_digit = True
                    break
            if found_second_last_digit:
                last_two_digits = int(last_two_digits_str)
                if last_two_digits % 25 == 0:
                    return 0
        elif n == 1 and num == '0':
            return 0
            
        if min_ops > n:
            return n
            
        return min_ops