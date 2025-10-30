from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        # In this problem, we “deliver” ones to a “target” position A.
        # Initially, if nums[A]==1, that one is picked up for free.
        # Then, using swapping moves (which cost 1 move each) and “flip” moves (which cost 1 move and are limited by maxChanges),
        # we want to “bring” additional ones from the left and right sides so that exactly k ones are eventually picked up.
        #
        # When bringing a one from one side, the mechanism is as follows:
        #   – The cell immediately adjacent to A on that side is the “delivery cell.”
        #   – If that cell already has a 1 (an “original” one) then one swap (cost 1) is needed.
        #   – If it is 0, one may be “supplied” from further away by a chain of swaps.
        #     In our cost model it turns out that if the very first neighbor isn’t “good,” we can “fix” it by flipping the delivery cell,
        #     costing 1 flip + 1 swap = 2 moves.
        #   – In fact, a brief simulation shows that, for a given “side” (left or right), the best possible cost is:
        #         • For a single delivery: cost = 1 move if the immediate neighbor is originally 1;
        #           otherwise you can get it in 2 moves – either by swapping a further‐away original one (if it happens to be at distance 2)
        #           or by “flipping” the cell.
        #         • For any additional delivery on the same side, no matter how far the original ones lie,
        #           an optimal strategy can “reset” the delivery by always fixing the same cell (the one adjacent to A).
        #           Thus each additional delivery can be done in 2 moves.
        #
        # In our solution we “model” the cost of delivering r ones from one side by a simple formula.
        # However, if an originally–good neighbor is available the cost is a little cheaper and no flip is needed.
        #
        # We define a helper function sideFunc(r, state) that returns (cost, flips) required to “deliver” r ones from one side.
        # Here state is an integer representing the “quality” of the side:
        #   0 means “both first cells are good” (i.e. the cell immediately adjacent and the next one are originally 1);
        #   1 means exactly one of the first two is 1;
        #   2 means neither is 1.
        #
        # We use the following formulas:
        #   • r == 0: (0, 0)
        #   • r == 1:
        #         state==0 --> (1, 0)  [best: immediate neighbor is 1 so only 1 swap]
        #         state==1 --> (2, 0)  [you can get it with 2 moves (swap chain) without a flip]
        #         state==2 --> (2, 1)  [you must flip the delivery cell]
        #   • For r >= 2:
        #         state==0 --> (2*r - 1, 0)
        #         state==1 --> (2*r, r - 1)   (i.e. aside from the first delivery, every extra one requires a flip if the “supply” isn’t perfect)
        #         state==2 --> (2*r, r)
        #
        # Then we “split” the required extra ones (after the possible initial pickup at A) between left and right.
        # For a chosen target A (in 0..n-1) we define:
        #    X = k – (1 if nums[A]==1 else 0)
        # and we must select L deliveries from the left (L between 0 and A) and R = X – L deliveries from the right (R ≤ n – A – 1).
        #
        # To “search” efficiently, notice that on each side the formulas are piecewise–linear.
        # Thus it suffices to check a few candidate splits (typically the endpoints of the valid interval together with a few nearby values).
        #
        # The overall answer is the minimum total moves over all valid choices of target A and splits L and R,
        # with the additional constraint that the total “flip” count (which comes from the sides)
        # does not exceed maxChanges.
        
        n = len(nums)
        INF = 10**18
        ans = INF

        # sideFunc returns (cost, flips) for r deliveries from one side, given the state.
        def sideFunc(r: int, state: int) -> (int, int):
            if r == 0:
                return (0, 0)
            if r == 1:
                if state == 0:
                    return (1, 0)
                elif state == 1:
                    return (2, 0)
                else:
                    return (2, 1)
            # r >= 2:
            if state == 0:
                return (2 * r - 1, 0)
            elif state == 1:
                return (2 * r, r - 1)
            else:
                return (2 * r, r)
        
        # For the left side (cells 0 .. A-1), define the side "state"
        # We'll set:
        #   state = 0 if both the immediate left cell and its neighbor are originally 1,
        #   state = 1 if exactly one of the two is 1,
        #   state = 2 if neither is 1.
        def getLeftState(A: int) -> int:
            if A <= 0:
                return 0  # no left side; not used.
            free1 = (A-1 >= 0 and nums[A-1] == 1)
            free2 = (A-2 >= 0 and nums[A-2] == 1)
            if free1 and free2:
                return 0
            elif free1 or free2:
                return 1
            else:
                return 2

        # Similarly, for the right side (cells A+1..n-1).
        def getRightState(A: int) -> int:
            if A >= n - 1:
                return 0
            free1 = (A+1 < n and nums[A+1] == 1)
            free2 = (A+2 < n and nums[A+2] == 1)
            if free1 and free2:
                return 0
            elif free1 or free2:
                return 1
            else:
                return 2

        # We iterate through all possible target positions A.
        for A in range(n):
            init = 1 if nums[A] == 1 else 0
            # Extra ones needed from the sides:
            X = k - init
            if X < 0:
                X = 0  # In case k==1 and nums[A]==1.
            # The maximum available ones from the sides: left side gives at most A, right at most (n-A-1).
            if X > A + (n - A - 1):
                continue  # Not possible from this A.
            # For a valid split L + R = X we must have:
            #   L is at least X - (n-A-1) (so that R <= n-A-1)
            #   and L is at most min(X, A)
            L_min = max(0, X - (n - A - 1))
            L_max = min(X, A)
            if L_min > L_max:
                continue  # no valid split.
            # Get the “state” for the left and right sides.
            left_state = 0 if A == 0 else getLeftState(A)
            right_state = 0 if A == n - 1 else getRightState(A)
            
            # Define helper lambdas for delivering r from left or right.
            def leftVal(r):
                if A == 0:
                    return (0, 0)
                return sideFunc(r, left_state)
            def rightVal(s):
                if A == n - 1:
                    return (0, 0)
                return sideFunc(s, right_state)
            
            # Because the formulas are piecewise–linear, it suffices to check a few candidate splits.
            cand_set = set()
            cand_set.add(L_min)
            cand_set.add(L_max)
            for cand in (0, 1, 2):
                if L_min <= cand <= L_max:
                    cand_set.add(cand)
            for cand in (X, X-1, X-2):
                if L_min <= cand <= L_max:
                    cand_set.add(cand)
            mid = (L_min + L_max) // 2
            if L_min <= mid <= L_max:
                cand_set.add(mid)
            
            # Evaluate each candidate split.
            for L_deliver in cand_set:
                R_deliver = X - L_deliver
                if R_deliver < 0 or R_deliver > (n - A - 1):
                    continue
                cost_left, flip_left = leftVal(L_deliver)
                cost_right, flip_right = rightVal(R_deliver)
                tot_cost = cost_left + cost_right
                tot_flips = flip_left + flip_right
                if tot_flips <= maxChanges:
                    ans = min(ans, tot_cost)
        
        return ans if ans < INF else -1