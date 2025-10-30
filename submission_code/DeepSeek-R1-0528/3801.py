class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def count_up_to(x):
            if x == 0:
                return 0
            digits = list(map(int, str(x)))
            L = len(digits)
            total = 0
            for s_val in range(1, 91):
                if s_val > 9 * L:
                    continue
                dp = [[[[0] * s_val for _ in range(s_val+1)] for __ in range(2)] for ___ in range(2)]
                init_prod = 1 % s_val
                dp[1][0][0][init_prod] = 1
                for i in range(L):
                    new_dp = [[[[0] * s_val for _ in range(s_val+1)] for __ in range(2)] for ___ in range(2)]
                    for tight in (0, 1):
                        for started in (0, 1):
                            for cs in range(s_val+1):
                                for cp in range(s_val):
                                    cnt = dp[tight][started][cs][cp]
                                    if cnt == 0:
                                        continue
                                    upper_bound = digits[i] if tight else 9
                                    for d in range(0, upper_bound+1):
                                        new_tight = tight and (d == upper_bound)
                                        if not started and d == 0:
                                            new_started = 0
                                            new_cs = cs
                                            new_cp = cp
                                            new_dp[new_tight][new_started][new_cs][new_cp] += cnt
                                        else:
                                            new_started = 1
                                            new_cs = cs + d
                                            if new_cs > s_val:
                                                continue
                                            new_cp = (cp * d) % s_val
                                            new_dp[new_tight][new_started][new_cs][new_cp] += cnt
                    dp = new_dp
                res_s = 0
                for tight in (0, 1):
                    res_s += dp[tight][1][s_val][0]
                total += res_s
            return total
        
        return count_up_to(r) - count_up_to(l-1)