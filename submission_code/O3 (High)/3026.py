class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10 ** 9 + 7
        
        # the largest integer that is strictly smaller than target / 2, plus the
        # middle value when target is even (it is safe to keep it once because
        # we are not allowed to store duplicates)
        m = target // 2                       # count of smallest integers we can always take
        
        # Case 1 : the first n natural numbers already satisfy the constraint
        if n <= m:
            return (n * (n + 1) // 2) % MOD
        
        # Case 2 : we take 1 … m first, then we still need k more numbers.
        # Those k numbers start from `target` because 
        # target ‑ m , … , target-1 are forbidden (they are complements of 1 … m).
        k = n - m
        
        sum_first_part  = m * (m + 1) // 2                    # sum of 1 … m
        sum_second_part = k * target + k * (k - 1) // 2       # sum of target … target+k-1
        
        return (sum_first_part + sum_second_part) % MOD