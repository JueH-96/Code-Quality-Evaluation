from typing import List

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        total = 0
        for l, r in queries:
            s = self.compute_sum(l, r)
            total += (s + 1) // 2
        return total
    
    def compute_sum(self, l: int, r: int) -> int:
        current_sum = 0
        a = 1  # Starting with k=1, a is 4^(k-1) = 4^0 = 1
        k = 1
        while a <= r:
            b = 4 * a - 1
            start = max(a, l)
            end = min(b, r)
            if start <= end:
                current_sum += (end - start + 1) * k
            a *= 4
            k += 1
        return current_sum