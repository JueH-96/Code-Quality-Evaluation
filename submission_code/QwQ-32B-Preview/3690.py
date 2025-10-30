class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        from itertools import groupby
        
        def is_possible(max_len):
            total_flips = 0
            for key, group in groupby(s):
                run_len = len(list(group))
                total_flips += run_len // (max_len + 1)
            return total_flips <= numOps
        
        n = len(s)
        low, high = 1, n
        while low < high:
            mid = (low + high) // 2
            if is_possible(mid):
                high = mid
            else:
                low = mid + 1
        return low