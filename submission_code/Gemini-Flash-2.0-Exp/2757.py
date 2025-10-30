class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def solve(num: str, min_sum: int, max_sum: int) -> int:
            n = len(num)
            dp = {}

            def dfs(idx: int, sum_so_far: int, is_tight: bool) -> int:
                if idx == n:
                    if min_sum <= sum_so_far <= max_sum:
                        return 1
                    else:
                        return 0
                
                if (idx, sum_so_far, is_tight) in dp:
                    return dp[(idx, sum_so_far, is_tight)]
                
                ans = 0
                upper_bound = int(num[idx]) if is_tight else 9
                
                for digit in range(upper_bound + 1):
                    new_is_tight = is_tight and (digit == upper_bound)
                    ans = (ans + dfs(idx + 1, sum_so_far + digit, new_is_tight)) % MOD
                
                dp[(idx, sum_so_far, is_tight)] = ans
                return ans

            return dfs(0, 0, True)

        ans2 = solve(num2, min_sum, max_sum)
        ans1 = solve(num1, min_sum, max_sum)

        sum_digits_num1 = sum(int(digit) for digit in num1)
        
        if min_sum <= sum_digits_num1 <= max_sum:
            return (ans2 - ans1 + 1 + MOD) % MOD
        else:
            return (ans2 - ans1 + MOD) % MOD