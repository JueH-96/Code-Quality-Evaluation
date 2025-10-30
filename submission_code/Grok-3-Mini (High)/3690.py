class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        # Find run lengths
        runs = []
        i = 0
        while i < n:
            curr_char = s[i]
            length = 0
            while i < n and s[i] == curr_char:
                length += 1
                i += 1
            runs.append(length)
        
        # Binary search for the minimum L
        left = 1
        right = n
        result = n  # Initialize to maximum possible length
        while left <= right:
            mid = (left + right) // 2
            # Compute total flips needed for max run length <= mid
            sum_flips = 0
            for run_len in runs:
                sum_flips += (run_len // (mid + 1))
            if sum_flips <= numOps:
                result = mid
                right = mid - 1  # Try to find a smaller L
            else:
                left = mid + 1  # Need a larger L
        return result