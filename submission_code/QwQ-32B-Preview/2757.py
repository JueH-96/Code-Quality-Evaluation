import functools

class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def str_to_digits(s):
            return [int(d) for d in s]
        
        def count_up_to(upper_limit_str):
            upper_limit = str_to_digits(upper_limit_str)
            memo = {}
            
            @functools.lru_cache(maxsize=None)
            def count_dp(pos, current_sum, is_less):
                if pos == len(upper_limit):
                    if min_sum <= current_sum <= max_sum:
                        return 1
                    else:
                        return 0
                if (pos, current_sum, is_less) in memo:
                    return memo[(pos, current_sum, is_less)]
                result = 0
                if is_less:
                    for d in range(10):
                        if current_sum + d > max_sum:
                            break
                        result = (result + count_dp(pos + 1, current_sum + d, True)) % MOD
                else:
                    for d in range(upper_limit[pos] + 1):
                        if d < upper_limit[pos]:
                            result = (result + count_dp(pos + 1, current_sum + d, True)) % MOD
                        else:
                            result = (result + count_dp(pos + 1, current_sum + d, False)) % MOD
                memo[(pos, current_sum, is_less)] = result
                return result
            
            return count_dp(0, 0, False)
        
        count_up_to_num2 = count_up_to(num2)
        
        num1_minus_one = str(int(num1) - 1)
        if int(num1) - 1 < 0:
            count_up_to_num1_minus_one = 0
        else:
            count_up_to_num1_minus_one = count_up_to(num1_minus_one)
        
        answer = (count_up_to_num2 - count_up_to_num1_minus_one) % MOD
        if answer < 0:
            answer += MOD
        return answer