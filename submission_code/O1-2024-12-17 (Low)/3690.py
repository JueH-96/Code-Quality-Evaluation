class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        
        # Compute lengths of consecutive runs of identical characters
        runs = []
        current_char = s[0]
        current_len = 1
        
        for i in range(1, n):
            if s[i] == current_char:
                current_len += 1
            else:
                runs.append(current_len)
                current_char = s[i]
                current_len = 1
        runs.append(current_len)  # append last run
        
        # Function to check if we can achieve all runs <= x
        # using at most numOps flips.  A run of length L needs
        # floor(L/(x+1)) flips to ensure sub-runs of length <= x.
        def can_achieve(x: int) -> bool:
            flips_needed = 0
            for length in runs:
                # Each flip will remove 1 character from the run
                # (because that character is flipped to the other bit)
                # effectively splitting the run and/or reducing it.
                # Minimal flips = sum of floor(run_length/(x+1))
                flips_needed += length // (x + 1)
                if flips_needed > numOps:
                    return False
            return flips_needed <= numOps
        
        # Binary search for the minimal x in [1..n]
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            if can_achieve(mid):
                right = mid
            else:
                left = mid + 1
        
        return left