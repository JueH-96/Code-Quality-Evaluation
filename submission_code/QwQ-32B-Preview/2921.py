class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def count_up_to(N):
            if N == "0":
                return 0
            N_list = [int(d) for d in N]
            len_N = len(N_list)
            
            # Count stepping numbers with length less than len(N)
            sum_length = sum(count_stepping_with_length(l) for l in range(1, len_N))
            
            # Count stepping numbers with length equal to len(N) and value <= N
            sum_length_and_value = count_stepping_with_length_and_value(len_N, N_list)
            
            return (sum_length + sum_length_and_value) % MOD

        @lru_cache(maxsize=None)
        def count_stepping_with_length(l):
            def dp(pos, prev_digit):
                if pos == l:
                    return 1
                if pos == 0:
                    sum = 0
                    for d in range(1, 10):
                        sum += dp(pos + 1, d)
                    return sum
                else:
                    sum = 0
                    for d in range(0, 10):
                        if abs(d - prev_digit) == 1:
                            sum += dp(pos + 1, d)
                    return sum
            return dp(0, -1)

        def count_stepping_with_length_and_value(l, N_list):
            @lru_cache(maxsize=None)
            def dp(pos, prev_digit, is_tight):
                if pos == l:
                    return 1
                if is_tight:
                    upper = N_list[pos]
                else:
                    upper = 9
                if pos == 0:
                    start = 1
                else:
                    start = 0
                sum = 0
                for d in range(start, upper + 1):
                    if pos > 0 and prev_digit != -1 and abs(d - prev_digit) != 1:
                        continue
                    next_tight = is_tight and d == upper
                    sum += dp(pos + 1, d, next_tight)
                return sum
            return dp(0, -1, 1)

        if low == "1" and high == "1":
            return 1
        if low == "1":
            low_minus_one = "0"
        else:
            low_minus_one = str(int(low) - 1)
        result = (count_up_to(high) - count_up_to(low_minus_one) + MOD) % MOD
        return result