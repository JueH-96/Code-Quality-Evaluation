import math

class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        # First, get the lengths of all contiguous blocks (runs) in s.
        run_lengths = []
        count = 1
        for i in range(1, n):
            if s[i] == s[i-1]:
                count += 1
            else:
                run_lengths.append(count)
                count = 1
        run_lengths.append(count)
        
        # Without any operations, the answer is simply the maximum run length.
        max_run = max(run_lengths)
        if numOps == 0:
            return max_run
        
        # The idea:
        # We want to guarantee that after at most numOps flips the longest run is at most L.
        # When you flip a bit inside a contiguous run, that flipped bit (now opposite) breaks the run.
        # In a run of length r, if you flip k bits inside (each flip removes that element
        # from the run of interest), then the "remaining" same-bit segments are of total length r - k.
        # These will be separated into (k+1) segments.
        # In order for every segment to be of length at most L, we need:
        #      (r - k) <= (k+1) * L.
        # Rearranging:
        #      r <= (L+1)*k + L.
        # Thus, for a given r > L we need the minimal integer k satisfying:
        #      k >= (r - L)/(L+1).
        # In other words, required flips = ceil((r - L) / (L+1))   (and zero if r <= L).
        #
        # Our strategy is to perform a binary search on L (the candidate for maximum
        # contiguous block length after operations) from 1 to max_run.
        
        def feasible(L: int) -> bool:
            total_flips = 0
            for r in run_lengths:
                if r > L:
                    # Calculate how many flips are needed for this run.
                    flips = math.ceil((r - L) / (L + 1))
                    total_flips += flips
                    if total_flips > numOps:
                        return False
            return total_flips <= numOps
        
        low, high = 1, max_run
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if feasible(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
                
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.minLength("000001", 1))  # Expected output: 2
    # Example 2:
    print(sol.minLength("0000", 2))    # Expected output: 1
    # Example 3:
    print(sol.minLength("0101", 0))    # Expected output: 1