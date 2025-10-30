class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        dp = [[[-1 for _ in range(400)] for _ in range(22)] for _ in range(2)]
        
        def digit_sum(n):
            return sum(int(digit) for digit in str(n))
        
        def count_good_numbers(idx, tight, sum_so_far):
            if idx == -1:
                return min_sum <= sum_so_far <= max_sum
            elif dp[tight][idx][sum_so_far] != -1:
                return dp[tight][idx][sum_so_far]
            else:
                res = 0
                end = 9 if not tight else int(num2[idx])
                for i in range(end + 1):
                    if sum_so_far + i > max_sum:
                        break
                    res = (res + count_good_numbers(idx - 1, tight & (i == end), sum_so_far + i)) % MOD
                dp[tight][idx][sum_so_far] = res
                return res
        
        num2 = num2.zfill(22)
        num1 = num1.zfill(22)
        total = count_good_numbers(21, 1, 0)
        num2 = num1
        total = (total - count_good_numbers(21, 1, 0) + MOD) % MOD
        return total