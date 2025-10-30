class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        # Number of conflicting pairs in [1, target-1]:
        # pairs are (1, target-1), (2, target-2), ..., up to ((target-1)//2, target-((target-1)//2))
        P = (target - 1) // 2
        # If target is even, the element target//2 is unpaired (since we can't pick it twice)
        has_mid = (target % 2 == 0)
        C = P + (1 if has_mid else 0)  # total safe picks < target
        
        # sum of 1..x = x*(x+1)//2
        def sum_first(x):
            return x * (x + 1) // 2
        
        # Case 1: we can fill all n from the smallest picks 1,2,...
        if n <= P:
            res = sum_first(n)
            return res % MOD
        
        # Sum of all the smaller picks 1..P
        sum_small = sum_first(P)
        
        # Case 2: target even and we pick the unpaired mid element as the (P+1)-th
        if has_mid and n == P + 1:
            res = sum_small + (target // 2)
            return res % MOD
        
        # Case 3: we pick all safe < target (including mid if any), then continue from target upward
        sum_all_safe = sum_small + (target // 2 if has_mid else 0)
        # remaining to pick
        m = n - C
        # we pick target, target+1, ..., target+m-1
        # sum = m*target + sum_first(m-1)
        sum_tail = m * target + sum_first(m - 1)
        
        res = sum_all_safe + sum_tail
        return res % MOD