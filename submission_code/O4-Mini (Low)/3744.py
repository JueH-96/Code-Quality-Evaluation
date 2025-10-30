from typing import List

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        # Precompute powers of 4 up to cover r <= 1e9
        p4 = [1]
        while p4[-1] <= 10**9:
            p4.append(p4[-1] * 4)
        
        # S(n) = sum_{x=1..n} f(x), where f(x) = number of times x //= 4 until becomes 0
        def S(n: int) -> int:
            if n <= 0:
                return 0
            total = 0
            # f(x) = k for x in [4^{k-1}, 4^k - 1]
            # k starts from 1
            # p4[k-1] = 4^{k-1}, p4[k] = 4^k
            max_k = len(p4)
            for k in range(1, max_k):
                low = p4[k-1]
                if low > n:
                    break
                high = p4[k] - 1
                cnt = min(n, high) - low + 1
                if cnt > 0:
                    total += cnt * k
            return total
        
        ans = 0
        for l, r in queries:
            s = S(r) - S(l - 1)
            # each operation picks two elements => 2 selections
            ops = (s + 1) // 2
            ans += ops
        return ans