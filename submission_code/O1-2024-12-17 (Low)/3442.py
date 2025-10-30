class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        """
        We want to find the maximum total x we can accumulate by repeatedly picking
        rewards rewardValues[i] that are strictly greater than our current sum x.
        
        Key idea:
          • We can pick rewards in any order.
          • A straightforward (but subtle) way to handle this is:
              - Sort the rewards in ascending order.
              - Maintain a set of all possible "current sums" that can occur.
              - Initially, the set of feasible sums is {0}.
              - For each reward r in ascending order, for each feasible sum s so far:
                if r > s, then we can form a new sum s + r.
              - Add s + r into our new set of feasible sums.
              - In the end, the answer is the maximum sum in the feasible set.

        Explanation on why it works:
          - Because we can pick the rewards in *any* order, sorting them only helps
            us systematically consider which new sums become reachable whenever a reward
            is larger than some feasible sum.
          - Critically, we do not *have* to pick a reward even if it's allowed. So at each
            reward, we keep all existing sums (to represent "skipping" that reward) plus
            any newly formed sums.
          - This set-based dynamic approach safely explores all ways to skip or pick
            rewards (in any sequence) without exponential recursion, though in the worst case
            the set of sums can grow quite large. For the given constraints it will
            still work in a reasonable time.

        Example walkthrough (rewardValues = [1, 6, 4, 3, 2]):
          • sorted => [1, 2, 3, 4, 6]
          • start feasible = {0}
            - r=1: from s=0 (1>0) => new sum = 1
              feasible = {0,1}
            - r=2: check s=0 => 2>0 => add 2
                     check s=1 => 2>1 => add 3
              feasible = {0,1,2,3}
            - r=3: check s=0 => 3>0 => add 3 (already in set)
                     check s=1 => 3>1 => add 4
                     check s=2 => 3>2 => add 5
                     check s=3 => 3>3 => not allowed
              feasible = {0,1,2,3,4,5}
            - r=4: check s=0 => 4>0 => add 4 (already in)
                     s=1 => 4>1 => add 5 (in)
                     s=2 => 4>2 => add 6
                     s=3 => 4>3 => add 7
                     s=4 => 4>4 => not allowed
                     s=5 => 4>5 => not allowed
              feasible = {0,1,2,3,4,5,6,7}
            - r=6: check s=0 => 6>0 => add 6 (in)
                     s=1 => 6>1 => add 7 (in)
                     s=2 => 6>2 => add 8
                     s=3 => 6>3 => add 9
                     s=4 => 6>4 => add 10
                     s=5 => 6>5 => add 11  <-- this is the key sum
                     s=6 => 6>6 => not allowed
                     s=7 => 6>7 => not allowed
                     s=8 => 6>8 => not allowed
                     etc.
              feasible = {0,1,2,3,4,5,6,7,8,9,10,11}

          max(feasible) = 11
          which matches the example's optimal answer.

        Complexity:
          - Let n = len(rewardValues).
          - In the worst case, the number of distinct sums in 'feasible' could, in principle,
            grow on the order of O(n^2). Each step we iterate over that set and possibly
            add new sums. So in practice this is roughly O(n * M), where M is the size
            of the set. This can be done within the constraints for n up to 2000.
        """
        import bisect
        
        # Sort the rewards in ascending order
        rewardValues.sort()
        
        feasible_sums = {0}
        
        for r in rewardValues:
            # We will build up a new set of sums that includes all old ones
            # plus any new sums formed by adding r, provided r > old_sum.
            new_sums = set(feasible_sums)
            for s in feasible_sums:
                if r > s:
                    new_sums.add(s + r)
            feasible_sums = new_sums
        
        return max(feasible_sums)