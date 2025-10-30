from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # Explanation:
        # Each element nums[i] can either be left as is or increased by 1.
        # Thus an element originally equal to x can be used to “cover”
        # the number x (if left unchanged) or x+1 (if increased).
        # We want to choose a consecutive set of integer targets,
        # say [a, a+1, …, a+L-1], and assign a distinct element (after possibly increasing)
        # to each target. An element originally x can cover target x or target x+1.
        #
        # A useful observation: For any target value t,
        # an element originally equal to t (non‐increased) and an element originally equal to t-1 (increased)
        # are the only ways to get t.
        # 
        # In our “assignment” view we want to “match” consecutive target numbers with indices.
        # When choosing an assignment we would prefer to “use up” an element that comes from the increased pool
        # (i.e. an element originally t-1) because it can be used only for target t,
        # while an element originally t is more flexible (it could be used for target t
        # or saved and used to cover target t+1 by increasing it).
        #
        # We can model a greedy matching on the target numbers (the “consecutive” numbers we want to cover)
        # in terms of a simple state variable.
        # At target x, we have two situations:
        #   • "Free" state: we already have a leftover element from the previous target (an element from x-1
        #     that has not been “wastefully” used) and so target x is automatically coverable. In that case,
        #     we do not consume any fresh element from the group of x's and we “carry” the new supply f(x) (the count
        #     of elements originally equal to x) forward.
        #   • "No‐free" state: we do not have a leftover.
        #     In that case target x must be covered by using one from the f(x) supply.
        #     But that “uses up” one element. So if f(x)==1, then no element is left (state stays False),
        #     while if f(x) >= 2, one unit is used and at least one remains (state becomes True).
        #
        # Let f(x) be the count (frequency) of values equal to x in the original array.
        # (Note: an element originally equal to x can also cover target x+1 if increased.)
        #
        # Define dp(x, free) for a target value x with a flag "free" (True means we have a leftover element
        # from the previous target):
        #   - If free is True, then target x is automatically covered.
        #     After covering x, we “add” the fresh supply f(x) (these are elements originally equal to x)
        #     which gives new free state = True if f(x) > 0, or False otherwise.
        #     Hence: dp(x, True) = 1 + dp(x+1, True) if f(x)>0, 1 + dp(x+1, False) if f(x)==0.
        #   - If free is False, target x must be covered using one element from the f(x) supply.
        #     If f(x)==0 then we cannot cover x and the chain stops.
        #     If f(x)>0 then we “use” one element.
        #     If f(x) >= 2 then one element remains and state becomes True for x+1,
        #     otherwise if f(x)==1 then state remains False.
        #     So: dp(x, False) = 0 if f(x)==0, else 1 + dp(x+1, True) if f(x) >= 2 or 1 + dp(x+1, False) if f(x)==1.
        #
        # Notice that in a valid assignment, the choice of starting target a is free.
        # However, the first target a can be covered in one of two ways:
        #   - Either by using an element originally equal to a-1 (increased to a) – which gives a free assignment;
        #   - Or by using an element originally equal to a.
        # We therefore “simulate” starting with state = True if f(a-1)>0 (so we have some element from a-1
        # available to cover a) or with state = False otherwise.
        #
        # We will precompute the dp values over the candidate target numbers.
        # Let m = max(nums) + 1.
        # We define f[x] for x going from 0 to m (with f[0]=0).
        # Then for x in [1, m]:
        #   dp1[x] := dp(x, True)
        #   dp0[x] := dp(x, False)
        # with base condition dp( m+1, free ) = 0.
        #
        # Recurrences (computed for x from m down to 1):
        #   if f[x] > 0:
        #       dp1[x] = 1 + dp1[x+1]   (because free state uses no fresh element, so state becomes True as f[x]>0)
        #   else:
        #       dp1[x] = 1 + dp0[x+1]   (if f[x] == 0, free state covers x then state becomes False)
        #
        #   dp0[x] = 0 if f[x] == 0
        #   if f[x] > 0:
        #       dp0[x] = 1 + (dp1[x+1] if f[x] >= 2 else dp0[x+1])
        #
        # For any starting candidate a (1 <= a <= m), the maximum chain length that can be covered is:
        #   chain(a) = dp( a, freeState )
        # where freeState = True if f[a-1] > 0, or False otherwise.
        # Then the answer is the maximum chain length over all such a.
        #
        # Implementation note: m = max(nums) + 1.
        # The dp values will be computed for x = 1,2,...,m (and dp[m+1] = 0).
        
        if not nums:
            return 0
        
        # Determine m = max(nums)+1. (Candidates are from 1 up to m.)
        max_num = max(nums)
        m = max_num + 1
        
        # Build frequency array f[0..m].
        # f[x] is the count of numbers originally equal to x.
        f = [0] * (m + 1)  # indices 0..m; note: no nums are 0 because nums[i] are positive.
        for num in nums:
            if num <= m:
                f[num] += 1
            else:
                # If an element is larger than max_num (should not happen) adjust m accordingly.
                # But given constraints this branch is not needed.
                pass
        
        # dp0[x] = dp(x, False), dp1[x] = dp(x, True)
        dp0 = [0] * (m + 2)
        dp1 = [0] * (m + 2)
        # Base condition: for x = m+1, dp = 0
        dp0[m+1] = 0
        dp1[m+1] = 0
        
        # Compute dp from x = m down to 1.
        # Note: f[x] is defined for x in 0..m.
        for x in range(m, 0, -1):
            if f[x] > 0:
                dp1[x] = 1 + dp1[x+1]
            else:
                dp1[x] = 1 + dp0[x+1]
            
            if f[x] == 0:
                dp0[x] = 0
            else:
                # if we have no free element, then we must use one element from f[x].
                # If f[x] >= 2, after using one, at least one remains so next state is free.
                # Otherwise (f[x]==1), next state remains not free.
                if f[x] >= 2:
                    dp0[x] = 1 + dp1[x+1]
                else:
                    dp0[x] = 1 + dp0[x+1]
        
        # Now consider every possible starting candidate a from 1 to m.
        # At the very beginning, to cover target a,
        # we can use an element originally equal to a-1 (i.e. if f[a-1] > 0, then we have "free" state),
        # or else we must use an element from f[a] (state False).
        best = 0
        for a in range(1, m+1):
            if a - 1 >= 0 and f[a-1] > 0:
                chain = dp1[a]
            else:
                chain = dp0[a]
            if chain > best:
                best = chain
        
        return best