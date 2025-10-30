class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        """
        We can increment each array element by at most 1, then choose a subset
        of final values that form a consecutive sequence (strictly increasing
        by exactly 1 when sorted). We want the size of the largest such subset.

        A succinct way to do this is:
        1) Sort the array.
        2) Keep a variable 'last' that tracks the last (largest) value we have
           placed in our consecutive sequence so far. Initialize it to one less
           than the first sorted element.
        3) Iterate through the sorted elements in ascending order. For each
           element x:
             - We want to see if we can place the next needed value (which is
               last+1) using x itself or x+1. That is possible exactly if
               last+1 is in [x, x+1].
             - If it fits, we increment our answer (the length of the formed
               consecutive sequence) and update last to last+1.
             - If it does not fit, we skip x and continue.

        Why this works:
        - Sorting lets us consider elements from smallest to largest so that
          we can "grow" a strictly increasing sequence steadily.
        - Starting 'last' at (smallest_element - 1) effectively means "we
          haven't yet placed any final value," so the first element can set
          the initial place in the consecutive run.
        - Each element can represent exactly one slot (either its own value or
          its value+1). By greedily filling the consecutive slot last+1 if
          possible, we ensure the sequence grows as long as it can, and this
          yields the maximum length of such a strictly consecutive choice.

        Example walkthrough:
          nums = [5,5,6]
          sorted = [5,5,6], last = 4, answer = 0
          - Look at x=5: last+1 = 5 in [5,6]? yes => answer=1, last=5
          - Look at x=5: last+1 = 6 in [5,6]? yes => answer=2, last=6
          - Look at x=6: last+1 = 7 in [6,7]? yes => answer=3, last=7
          So we get 3, corresponding to final values [5,6,7].

          nums = [2,1,5,1,1] (from the problem statement)
          sorted = [1,1,1,2,5], last = 0, answer = 0
          - x=1: last+1=1 in [1,2]? yes => answer=1, last=1
          - x=1: last+1=2 in [1,2]? yes => answer=2, last=2
          - x=1: last+1=3 in [1,2]? no => skip
          - x=2: last+1=3 in [2,3]? yes => answer=3, last=3
          - x=5: last+1=4 in [5,6]? no => skip
          answer=3, which matches the example.

        Time complexity is O(n log n) due to sorting. The single pass after sort
        is O(n). This fits well within the constraints (up to 10^5 elements).
        """
        nums.sort()
        # 'last' is the final value we last placed in the consecutive sequence
        # Start it one less than the smallest element so we can begin filling
        # from that smallest element if possible.
        last = nums[0] - 1
        answer = 0

        for x in nums:
            # We want to place last+1 if possible.
            needed = last + 1
            # Check if x can become 'needed': that is needed in [x, x+1].
            if x <= needed <= x + 1:
                answer += 1
                last = needed  # We have placed the next consecutive value

        return answer