class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        """
        We need to find two indices i and j such that:
            |i - j| >= x
        and the absolute difference |nums[i] - nums[j]| is minimized.

        --------------------------------------------------------------------
        APPROACH:

        1) Special Case (x == 0):
           In this case, any distinct pair of indices is valid. The classic
           way to find the minimum absolute difference among all pairs is:
               - sort the array
               - scan adjacent elements in the sorted order
               - track the minimum difference among those adjacent elements
           The minimal difference among all pairs emerges between some pair
           of adjacent values in the sorted array.

        2) General Case (x > 0):
           We will sweep through the array from left to right while maintaining
           a balanced data structure (SortedList) of values that are "far enough"
           behind the current index i (i.e., those at indices ≤ i - x).

           Concretely:
             - Let S be a balanced structure (e.g., SortedList) initially empty.
             - For i in [0..n-1]:
                 (a) If i >= x, insert nums[i - x] into S.  This ensures that S
                     contains exactly those elements whose indices are at least x
                     behind i.
                 (b) If S is not empty, binary-search in S for nums[i] to find the
                     closest value(s) and update the answer with the smaller of
                     those differences.

           This process takes O(n log n) time provided we have a balanced data
           structure that supports both insertion and finding neighbors in O(log n).
           In Python, one convenient choice is "SortedList" from the "sortedcontainers"
           library. (If that library is not allowed in a particular environment,
           one would need to implement a balanced BST or another data structure
           strategy.)

        --------------------------------------------------------------------
        COMPLEXITY:

        - For x = 0, sorting takes O(n log n). Scanning adjacent pairs then
          takes O(n).
        - For x > 0, each insertion and neighbor search is O(log n) using
          SortedList (or a well-implemented balanced tree). Doing this n times
          leads to O(n log n) total.

        --------------------------------------------------------------------
        EXAMPLES:

        Example 1:
            nums = [4,3,2,4], x = 2
            Output = 0
            Explanation: Pair (index 0=4, index 3=4) => difference = 0, which
            is minimal.

        Example 2:
            nums = [5,3,2,10,15], x = 1
            Output = 1
            Explanation: Pair (index 1=3, index 2=2) => difference = 1.

        Example 3:
            nums = [1,2,3,4], x = 3
            Output = 3
            Explanation: Pair (index 0=1, index 3=4) => difference = 3.

        --------------------------------------------------------------------
        """

        import math

        # If x == 0, we can find the minimum difference among all pairs by sorting.
        if x == 0:
            sorted_nums = sorted(nums)
            ans = math.inf
            for i in range(len(sorted_nums) - 1):
                ans = min(ans, sorted_nums[i+1] - sorted_nums[i])
            return ans

        # Otherwise (x > 0), use a balanced data structure to enforce |i - j| >= x.
        # If you have "sortedcontainers" library available:
        # from sortedcontainers import SortedList
        #
        # Otherwise, here is a fallback with bisect + list insertion (O(n^2) worst-case).
        # For large n, this may be too slow in Python unless "sortedcontainers" is allowed.
        import bisect

        s = []  # We'll keep a sorted list of values that are at least x behind the current index.
        ans = math.inf

        for i in range(len(nums)):
            # As soon as i >= x, nums[i - x] becomes valid to pair with nums[i].
            if i >= x:
                bisect.insort_left(s, nums[i - x])

            # Now find the closest value(s) in s to nums[i].
            if s:
                pos = bisect.bisect_left(s, nums[i])
                if pos < len(s):
                    ans = min(ans, abs(nums[i] - s[pos]))
                if pos > 0:
                    ans = min(ans, abs(nums[i] - s[pos - 1]))

        return ans