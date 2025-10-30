class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n - k
        total_or = 0
        for a in nums:
            total_or |= a
        
        def feasible(x):
            s = 0
            cur = (1 << 60) - 1
            for a in nums:
                cur &= a
                if (cur | x) == x:
                    s += 1
                    if s >= m:
                        break
                    cur = (1 << 60) - 1
            return s >= m
        
        ans = total_or
        for bit in range(29, -1, -1):
            if ans & (1 << bit):
                candidate = ans & ~(1 << bit)
                if feasible(candidate):
                    ans = candidate
        return ans