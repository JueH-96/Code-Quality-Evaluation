class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        res = []
        for l, r in queries:
            # Compute k_max for r
            k_max = 0
            x = r
            while x > 0:
                x = x // 4
                k_max += 1
            
            sum_steps = 0
            prev_power = 1  # 4^(k-1)
            for k in range(1, k_max + 1):
                lower = prev_power
                upper = (prev_power * 4) - 1
                a = max(l, lower)
                b = min(r, upper)
                if a <= b:
                    cnt = b - a + 1
                    sum_steps += cnt * k
                prev_power *= 4
            
            # Calculate the answer for this query
            ans = max(k_max, (sum_steps + 1) // 2)
            res.append(ans)
        
        return res