class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        # Precompute the A and B values for k from 1 to 16
        ab = []
        for k in range(1, 17):
            a = 4 ** (k - 1)
            b = (4 ** k) - 1
            ab.append((a, b))
        
        res = 0
        for l, r in queries:
            sum_s = 0
            for idx in range(len(ab)):
                a, b = ab[idx]
                if a > r:
                    break
                low = max(l, a)
                high = min(r, b)
                if low > high:
                    continue
                cnt = high - low + 1
                k_val = idx + 1
                sum_s += k_val * cnt
            res += (sum_s + 1) // 2
        return res