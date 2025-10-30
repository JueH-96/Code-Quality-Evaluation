class Solution:
    def earliestSecondToMarkIndices(self, nums: list, changeIndices: list) -> int:
        n, m = len(nums), len(changeIndices)
        # Build list of occurrences for indices 1 through n.
        occurrences = { i: [] for i in range(1, n+1) }
        for s, idx in enumerate(changeIndices, start=1):
            occurrences[idx].append(s)
        
        # If some index never appears in changeIndices, it's impossible.
        for i in range(1, n+1):
            if not occurrences[i]:
                return -1

        total_decrements = sum(nums)
        # A necessary condition is that the total available decrement operations (T - n)
        # is at least sum(nums). Also, you cannot mark an index i before its first occurrence.
        # Thus a valid candidate T must be at least:
        lb_candidate = n + total_decrements
        first_occ = max(occurrences[i][0] for i in range(1, n+1))
        L = max(lb_candidate, first_occ)
        if L > m:
            return -1  # because T is at most m

        # We'll binary search for the smallest T (in [L, m]) that is feasible.
        from bisect import bisect_right

        def feasible(T: int) -> bool:
            # Check enough total decrement steps.
            if T - n < total_decrements:
                return False
            caps = []
            for i in range(1, n+1):
                occ = occurrences[i]
                # Find the rightmost occurrence s from occ such that s <= T.
                pos = bisect_right(occ, T)
                if pos == 0:  # no available marking second for index i by time T.
                    return False
                candidate = occ[pos - 1]
                # capacity = candidate s - the number of decrements required for i.
                cap = candidate - nums[i-1]  # (nums is 0-indexed)
                if cap < 1:
                    return False
                caps.append(cap)
            caps.sort()
            # Check if the kth smallest capacity is at least k (marking slots are 1-indexed).
            for j, cap in enumerate(caps, start=1):
                if cap < j:
                    return False
            return True

        ans = -1
        lo, hi = L, m
        while lo <= hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans

# Example testing
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    # nums = [2,2,0], changeIndices = [2,2,2,2,3,2,2,1]
    # Expected output: 8
    print(sol.earliestSecondToMarkIndices([2,2,0], [2,2,2,2,3,2,2,1]))
    
    # Example 2:
    # nums = [1,3], changeIndices = [1,1,1,2,1,1,1]
    # Expected output: 6
    print(sol.earliestSecondToMarkIndices([1,3], [1,1,1,2,1,1,1]))
    
    # Example 3:
    # nums = [0,1], changeIndices = [2,2,2]
    # Expected output: -1
    print(sol.earliestSecondToMarkIndices([0,1], [2,2,2]))