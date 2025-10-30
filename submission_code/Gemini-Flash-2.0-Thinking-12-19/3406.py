class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        n = zero + one
        MOD = 10**9 + 7
        dp = {} # key: (length, zeros_count, ones_count, last_digit_run, last_digit) value: count
        
        def get_dp_value(length, zeros_count, ones_count, last_digit_run, last_digit):
            if (length, zeros_count, ones_count, last_digit_run, last_digit) in dp:
                return dp[(length, zeros_count, ones_count, last_digit_run, last_digit)]
            return 0
            
        if zero > 0:
            dp[(1, 1, 0, 1, 0)] = 1
        if one > 0:
            dp[(1, 0, 1, 1, 1)] = 1
            
        for length in range(2, n + 1):
            for zeros_count in range(zero + 1):
                for ones_count in range(one + 1):
                    if zeros_count + ones_count != length:
                        continue
                    for run_length in range(1, limit + 1):
                        # Placing 0 at current position
                        if zeros_count > 0:
                            count0 = 0
                            if run_length > 1:
                                count0 = (count0 + get_dp_value(length - 1, zeros_count - 1, ones_count, run_length - 1, 0)) % MOD
                            else: # run_length == 1
                                sum_prev_ones = 0
                                for prev_run_length in range(1, limit + 1):
                                    sum_prev_ones = (sum_prev_ones + get_dp_value(length - 1, zeros_count - 1, ones_count, prev_run_length, 1)) % MOD
                                count0 = (count0 + sum_prev_ones) % MOD
                            if count0 > 0:
                                dp[(length, zeros_count, ones_count, run_length, 0)] = (dp.get((length, zeros_count, ones_count, run_length, 0), 0) + count0) % MOD

                        # Placing 1 at current position
                        if ones_count > 0:
                            count1 = 0
                            if run_length > 1:
                                count1 = (count1 + get_dp_value(length - 1, zeros_count, ones_count - 1, run_length - 1, 1)) % MOD
                            else: # run_length == 1
                                sum_prev_zeros = 0
                                for prev_run_length in range(1, limit + 1):
                                    sum_prev_zeros = (sum_prev_zeros + get_dp_value(length - 1, zeros_count, ones_count - 1, prev_run_length, 0)) % MOD
                                count1 = (count1 + sum_prev_zeros) % MOD
                            if count1 > 0:
                                dp[(length, zeros_count, ones_count, run_length, 1)] = (dp.get((length, zeros_count, ones_count, run_length, 1), 0) + count1) % MOD
                                
        result = 0
        for run_length in range(1, limit + 1):
            result = (result + get_dp_value(n, zero, one, run_length, 0)) % MOD
            result = (result + get_dp_value(n, zero, one, run_length, 1)) % MOD
            
        return result