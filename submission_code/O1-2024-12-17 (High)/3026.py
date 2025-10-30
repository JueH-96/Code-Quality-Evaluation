class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        # Helper function: sum of 1..x, modulo MOD
        def sum_1_to_x(x):
            return (x * (x + 1) // 2) % MOD
        
        # For target >= 1:
        # Let M = floor((target-1)/2) if target is odd,
        #         = target//2         if target is even.
        # We can pick at most M numbers from 1..(target-1) in such a way
        # that no two chosen numbers sum to target.
        #
        # If n <= M:
        #    we can simply pick the first n positive integers (1..n),
        #    because their largest possible sum of any pair is 2n,
        #    which is less than target (since 2n <= 2M <= target if even,
        #    or 2n+1 <= 2M+1 = target if odd).
        #
        # Otherwise, we pick all M numbers from 1..(target-1) that do not conflict
        # (which effectively are 1..M), and then we still need (n - M) more numbers.
        # Those can all come from [target, target+1, ... ], since picking numbers >= target
        # cannot create a sum of exactly target with any already-chosen positive integer.
        #
        # Sum(1..M) = M*(M+1)//2
        # Sum of the first K integers starting at T is K*T + (K*(K-1))//2.
        
        if target == 1:
            # If target == 1, M = 0 below, so we'll end up in the "else" branch,
            # which also works correctly. But it may be clearer to let the logic flow.
            pass
        
        # Compute M
        if target % 2 == 0:
            M = target // 2
        else:
            M = (target - 1) // 2
        
        # If we only need n <= M, pick 1..n
        if n <= M:
            return sum_1_to_x(n)
        
        # Otherwise, pick 1..M, then pick (n - M) numbers from [target, target+1, ...]
        part1 = sum_1_to_x(M)  # sum of 1..M
        required = n - M
        # sum of target..(target + required - 1)
        # = required * target + (required*(required-1))//2
        part2 = (required * target) % MOD
        part2 = (part2 + (required * (required - 1) // 2)) % MOD
        
        return (part1 + part2) % MOD