class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        """
        We want to form a sequence of picks (in any order) subject to:
           - We start with x = 0
           - Each picked reward v must satisfy v > x, then x += v
           - An item can be picked at most once
        Our goal is to maximize the final x.

        --------------------------------------------------------------------
        OBSERVATION / APPROACH:

        If we think in terms of "which sums are achievable?" then each new
        reward v can only be added to those sums s which are strictly less
        than v (because we require v > current_sum).

        In other words:
           - Start with feasible = {0} meaning sum=0 is initially achievable.
           - Sort the rewardValues in non-decreasing order.
           - For each reward v in ascending order:
                * We look at all feasible sums that are strictly < v.
                * For each such feasible sum s, we can form new sum s+v.
                * Merge these new sums into the set of feasible sums.

        Why does sorting not lose generality? Because we are free to pick the
        rewards in any order we like: the only constraint is “next item > x”.
        By processing rewards from small to large and only extending sums from
        those strictly less than v at each step, we effectively simulate all
        valid ways of forming a chain of picks subject to v > current_sum.

        Finally, the answer is simply the maximum achievable sum in "feasible".

        EXAMPLE 1:
            rewardValues = [1,1,3,3]
            sort => [1,1,3,3]
            Initially feasible = [0]

            1st v=1:
              sums < 1 is [0]
              new sums = [0+1=1]
              feasible => [0,1]
            2nd v=1:
              sums < 1 is [0]
              new sums = [1]
              feasible => [0,1]; (merging 1 does not enlarge)
            3rd v=3:
              sums < 3 is [0,1, (not 2—since not in feasible)]
              new sums = [3,4]
              feasible => [0,1,3,4]
            4th v=3:
              sums < 3 is [0,1]
              new sums = [3,4]
              merging => feasible => [0,1,3,4]
            max(feasible)=4

        EXAMPLE 2:
            rewardValues = [1,6,4,3,2]
            sort => [1,2,3,4,6]
            feasible = [0]

            v=1:
              <1 => [0], new=[1], feasible=[0,1]
            v=2:
              <2 => [0,1], new=[2,3], feasible=[0,1,2,3]
            v=3:
              <3 => [0,1,2], new=[3,4,5], feasible=[0,1,2,3,4,5]
            v=4:
              <4 => [0,1,2,3], new=[4,5,6,7], feasible=[0,1,2,3,4,5,6,7]
            v=6:
              <6 => [0,1,2,3,4,5], new=[6,7,8,9,10,11]
              feasible=[0,1,2,3,4,5,6,7,8,9,10,11]
            max=11

        This matches the example.

        --------------------------------------------------------------------
        COMPLEXITY:

        - Let n = len(rewardValues).
        - In the worst case, each iteration can at most double the size of the
          feasible set (because we add at most as many new sums as feasible[:idx]).
        - However, because rewardValues[i] ≤ 2000, sums grow quickly, and the
          condition “< v” often prevents large branching. In practice, for n=2000,
          this remains feasible with a reasonable merge approach.

        We'll implement this carefully, keeping a sorted list of feasible sums
        and merging the new sums each time.

        --------------------------------------------------------------------
        IMPLEMENTATION DETAILS:

        1) Sort rewardValues.
        2) Maintain a sorted list "feasible" of achievable sums.
        3) For each v in ascending order:
             - Find how many sums in feasible are strictly < v (via bisect).
             - Generate those new sums by adding v to each of them.
             - Merge back into feasible (in O(len(feasible)) time).
        4) Answer is the last element of feasible (the maximum).
        """

        import bisect

        # Sort the input
        arr = sorted(rewardValues)

        # "feasible" will be a sorted list of unique sums we can achieve.
        feasible = [0]  # initially only sum=0 is achievable

        for v in arr:
            # Find how many sums in `feasible` are strictly < v
            idx = bisect.bisect_left(feasible, v)
            if idx <= 0:
                # No existing sum is strictly less than v, so we can't pick v now
                continue

            # We'll form new sums by adding v to each feasible sum that is < v
            # (i.e. feasible[:idx]).
            new_sums = []
            for s in feasible[:idx]:
                new_sums.append(s + v)

            # Merge new_sums into feasible (both are sorted)
            # We can do a classic 2-pointer merge; also remove duplicates.
            merged = []
            i = 0
            j = 0
            len_f = len(feasible)
            len_n = len(new_sums)

            while i < len_f and j < len_n:
                if feasible[i] < new_sums[j]:
                    if not merged or merged[-1] != feasible[i]:
                        merged.append(feasible[i])
                    i += 1
                elif new_sums[j] < feasible[i]:
                    if not merged or merged[-1] != new_sums[j]:
                        merged.append(new_sums[j])
                    j += 1
                else:
                    # They are equal
                    if not merged or merged[-1] != feasible[i]:
                        merged.append(feasible[i])
                    i += 1
                    j += 1

            # Append any leftover from feasible
            while i < len_f:
                if not merged or merged[-1] != feasible[i]:
                    merged.append(feasible[i])
                i += 1

            # Append leftover from new_sums
            while j < len_n:
                if not merged or merged[-1] != new_sums[j]:
                    merged.append(new_sums[j])
                j += 1

            feasible = merged

        return feasible[-1]  # The largest achievable sum