class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        # First, decompose the string into runs of consecutive identical bits.
        n = len(s)
        if n == 0:
            return 0

        runs = []
        curr_char = s[0]
        cnt = 1
        for i in range(1, n):
            if s[i] == curr_char:
                cnt += 1
            else:
                runs.append(cnt)
                curr_char = s[i]
                cnt = 1
        runs.append(cnt)
        
        # A helper function: given a run of length r and a target maximum block length L,
        # what is the minimum number of flips needed (applied inside that run) so that 
        # after breaking it with flips, every contiguous block of original (unchanged) bits has length <= L.
        # When you flip a bit inside a run, that flipped bit now is of the opposite value and
        # does not continue the run. With t flips inside a run of length r, you'll break it into (t+1)
        # blocks, but note that the flipped bits are "free" gaps. However, note that flipping reduces
        # the number of same-bit characters available (r - t remain). They will be partitioned into t+1 segments.
        # We want:
        #      ceil( (r - t) / (t+1) ) <= L.
        #
        # Rearranging a necessary condition, we want:
        #      (r - t) <= L*(t+1)
        # which is equivalent to:
        #      r <= t*(L+1) + L.
        #
        # So a minimal t that satisfies t >= (r - L) / (L+1) is enough, provided r > L.
        # Thus, if r <= L, no flips are needed.
        # Otherwise, the minimum flips required for that run is:
        #      flips_needed = ceil( (r - L) / (L+1) )
        import math
        def flips_for_run(r: int, L: int) -> int:
            if r <= L:
                return 0
            return math.ceil((r - L) / (L + 1))
        
        # We want to find the smallest L (at least 1) such that total flips needed over all runs
        # is <= numOps.
        # The maximum current run length is our upper bound (because with enough flips, we could
        # break even the longest run to have max block length 1).
        low, high = 1, max(runs)
        ans = high  # worst-case answer
        
        while low <= high:
            mid = (low + high) // 2
            total_flips = 0
            for r in runs:
                total_flips += flips_for_run(r, mid)
            if total_flips <= numOps:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans

# You can run some tests here:
if __name__ == '__main__':
    sol = Solution()
    # Example 1
    s1 = "000001"
    numOps1 = 1
    print(sol.minLength(s1, numOps1))  # Expected output: 2

    # Example 2
    s2 = "0000"
    numOps2 = 2
    print(sol.minLength(s2, numOps2))  # Expected output: 1

    # Example 3
    s3 = "0101"
    numOps3 = 0
    print(sol.minLength(s3, numOps3))  # Expected output: 1