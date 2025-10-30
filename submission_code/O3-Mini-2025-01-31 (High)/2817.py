class Solution:
    def minimumCost(self, s: str) -> int:
        # In‐this–solution we show that any sequence of prefix/suffix flips
        # (each flip is “invert from index 0 to i” costing i+1 or “invert from i to n–1” costing n–i)
        # that converts s into a uniform string is “equivalent” (modulo the order of operations)
        # to an assignment of operations at only those “boundaries” where the “need to flip”
        # changes.
        #
        # More precisely, if we define for a chosen target (either '0' or '1')
        # an array b with:
        #    b[i] = 1 if s[i] != target, and 0 otherwise,
        # then the desired final state is that every position i gets flipped if and only if b[i]==1.
        #
        # A short calculation (by “differencing” the flip‐equations) shows that the only positions where
        # we get to choose an operation “freely” (at minimum extra cost) are the indices i (0 <= i < n–1)
        # for which b[i] and b[i+1] differ.
        # For each such index i the “difference” may be achieved in one of two ways:
        #   Option A: perform a prefix–flip ending at i, cost = i+1. (This “adds” a 1 in our parity sum.)
        #   Option B: perform a suffix–flip starting at i+1, cost = n–i–1. (This “adds” a 0.)
        #
        # In any optimal solution the “transitions” (that is, the boundaries between
        # adjacent indices where b changes value) will be “handled” exactly once.
        #
        # However, the boundary equations (that is, what happens at index 0 and n–1) force a parity condition.
        # One may show that if we let a decision variable a_i (equal to 1 when we choose Option A on
        # the i–th boundary and 0 when we choose Option B) then a necessary condition is that:
        # 
        #      (x[n–1] + y[0] + sum_{i in boundaries} a_i) mod 2 = b[0]
        #
        # and also (because the “other end” has a similar equation)
        # 
        #      (x[n–1] + y[0] + (number_of_boundaries – sum a_i)) mod 2 = b[n–1].
        #
        # Here x[n–1] and y[0] refer to our extra freedom at the boundaries:
        # We may “pay extra” (a cost of n) to choose one extra flip operation covering the whole string 
        # (either a prefix–flip ending at n–1, or a suffix–flip starting at 0).
        #
        # In our code below we “simulate” the following:
        #  – For each adjacent pair i,i+1 where b changes (we call these transition boundaries),
        #    we have two choices:
        #       Option A: cost = i+1, “adds” 1 to our parity sum.
        #       Option B: cost = n–i–1, “adds” 0.
        #    We want to assign these choices so that the sum (mod 2) equals b[0]
        #    (if no extra boundary operation is used) or equals 1–b[0]
        #    (if we pay an extra cost of n at the boundary).
        #
        #  – Because the decisions are “independent” (except for the parity constraint)
        #    the unconstrained optimum is to choose on each boundary the cheaper option.
        #    If that “default” assignment does not satisfy the parity condition,
        #    we flip exactly one of the choices – picking the one that “costs extra” as little as possible.
        #
        #  – One may have a “free” boundary if both options cost equally.
        #    In that case the parity may be adjusted for free.
        #
        #  – There is one more twist: if there are no boundaries at all (i.e. s is already constant
        #    with respect to the difference from target) then if b[0]==1 (i.e. every bit is “wrong”)
        #    we must flip the entire string in one shot (with cost n).
        #
        # Finally we choose the best cost among target = '0' and target = '1'.
        
        n = len(s)
        # For a string of length 1, it is already uniform.
        if n == 1:
            return 0

        # Helper function:
        # Given a target value (either '0' or '1'),
        # compute the minimum cost (using the above approach) to transform s into a string
        # where every character equals target.
        def cost_for_target(target: str) -> int:
            # b[i] = 1 if s[i] != target; else 0.
            b = [1 if ch != target else 0 for ch in s]
            # Gather indices where b changes from one position to the next.
            trans_indices = []
            for i in range(n - 1):
                if b[i] != b[i+1]:
                    trans_indices.append(i)
            m = len(trans_indices)
            # If there are no boundaries, b is constant.
            # If b[0] is already 0, no flip is required;
            # if b[0]==1 (all bits are “wrong”), we must flip the entire string.
            if m == 0:
                return 0 if b[0] == 0 else n
            
            # For each boundary indexed by i in trans_indices:
            #   Option A (prefix flip ending at i): costA = i+1, contributes a 1.
            #   Option B (suffix flip starting at i+1): costB = n - i - 1, contributes a 0.
            unconstrained_cost = 0
            fixed_parity = 0  # Sum (mod 2) over boundaries where one choice is strictly cheaper.
            extra_cost_candidates = []  # Extra cost needed to flip a decision.
            free_exists = False  # True if there is any boundary where both options cost equally.
            
            for i in trans_indices:
                costA = i + 1
                costB = n - i - 1
                if costA == costB:
                    unconstrained_cost += costA  # (Same as costB.)
                    free_exists = True
                    # No fixed parity from such a boundary.
                elif costA < costB:
                    unconstrained_cost += costA
                    fixed_parity = (fixed_parity + 1) % 2  # Default decision gives 1.
                    extra_cost_candidates.append(costB - costA)
                else:  # costB < costA
                    unconstrained_cost += costB
                    # Default decision gives 0.
                    extra_cost_candidates.append(costA - costB)
            
            # Two scenarios are possible:
            #   Scenario 0: Do NOT use an extra whole–string operation (boundary cost 0).
            #       Then the parity of the decisions must equal b[0].
            #   Scenario 1: Use one extra boundary flip (cost n) so that the parity requirement is flipped:
            #       then the parity must equal 1 - b[0].
            P0 = b[0]         # Required parity if no boundary op is used.
            P1 = 1 - b[0]     # Required parity if we pay n extra.
            
            # Decide how much extra cost (if any) is needed to adjust the parity.
            def parity_adjust_cost(required, fixed_parity, extra_cost_candidates, free_exists):
                if free_exists:
                    # At a boundary where costs are equal we may choose arbitrarily.
                    return 0
                if fixed_parity % 2 == required:
                    return 0
                else:
                    if extra_cost_candidates:
                        return min(extra_cost_candidates)
                    else:
                        return float('inf')
            
            extra0 = parity_adjust_cost(P0, fixed_parity, extra_cost_candidates, free_exists)
            total0 = unconstrained_cost + extra0  # No extra cost at the boundaries.
            extra1 = parity_adjust_cost(P1, fixed_parity, extra_cost_candidates, free_exists)
            total1 = unconstrained_cost + extra1 + n  # Plus cost n for the extra flip.
            return min(total0, total1)
        
        # Try both targets and take the minimum.
        cost0 = cost_for_target('0')
        cost1 = cost_for_target('1')
        return min(cost0, cost1)


# --- Testing the solution with the given examples ---
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumCost("0011"))    # Expected output: 2
    print(sol.minimumCost("010101"))  # Expected output: 9

    # Some extra tests:
    print(sol.minimumCost("00"))      # Expected output: 0 (already uniform)
    print(sol.minimumCost("10"))      # Expected output: 1
    print(sol.minimumCost("01010"))   # Additional test case