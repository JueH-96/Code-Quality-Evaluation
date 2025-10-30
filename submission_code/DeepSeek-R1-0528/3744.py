class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        intervals = []
        for k in range(0, 17):
            start = 4 ** k
            end = 4 ** (k + 1) - 1
            intervals.append((start, end, k + 1))
        
        total_ops = 0
        for l, r in queries:
            total_hits = 0
            for start, end, steps_val in intervals:
                if start > r:
                    break
                low = max(l, start)
                high = min(r, end)
                if low <= high:
                    total_hits += (high - low + 1) * steps_val
            total_ops += (total_hits + 1) // 2
        return total_ops