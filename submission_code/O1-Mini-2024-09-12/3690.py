class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        
        def is_possible(max_len):
            total_flips = 0
            i = 0
            while i < n:
                run_length = 1
                while i + 1 < n and s[i] == s[i + 1]:
                    run_length += 1
                    i += 1
                flips_needed = run_length // (max_len + 1)
                total_flips += flips_needed
                i += 1
            return total_flips <= numOps
        
        left, right = 1, n
        ans = n
        while left <= right:
            mid = (left + right) // 2
            if is_possible(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans