class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        total_cost = 0

        # The goal is to make all characters in the string equal.
        # This is equivalent to making all adjacent characters s[i] and s[i+1] equal for 0 <= i < n-1.
        # Consider the difference between s[i] and s[i+1].
        # This difference exists initially if s[i] != s[i+1].
        # To make s[i] and s[i+1] equal in the final string, the relative state of this pair must be flipped
        # an odd number of times compared to their initial state.
        # The operations that flip the relative state of s[i] and s[i+1] are:
        # 1. Operation 1 at index i: Inverts s[0...i]. This flips s[i] but not s[i+1]. Cost: i + 1.
        # 2. Operation 2 at index i+1: Inverts s[i+1...n-1]. This flips s[i+1] but not s[i]. Cost: n - (i + 1).
        # Let x_k = 1 if Operation 1 at index k is applied an odd number of times, 0 otherwise.
        # Let y_k = 1 if Operation 2 at index k is applied an odd number of times, 0 otherwise.
        # The relative state s[i] vs s[i+1] is flipped if and only if x_i XOR y_{i+1} = 1.
        # If initially s[i] != s[i+1], we need x_i XOR y_{i+1} = 1 to make them equal in the end.
        # If initially s[i] == s[i+1], we need x_i XOR y_{i+1} = 0 to keep them equal in the end.
        # Let d_i = 1 if s[i] != s[i+1] initially, and 0 otherwise. We need x_i XOR y_{i+1} = d_i for 0 <= i < n-1.

        # The cost associated with the parity choice (x_i, y_{i+1}) to satisfy x_i XOR y_{i+1} = d_i is:
        # x_i * cost(Op1 at i) + y_{i+1} * cost(Op2 at i+1) = x_i * (i + 1) + y_{i+1} * (n - (i + 1)).

        # Consider the constraint x_i XOR y_{i+1} = d_i for a specific i.
        # The variables involved are x_i and y_{i+1}.
        # The constraint for the previous boundary i-1 (if i > 0) is x_{i-1} XOR y_i = d_{i-1}, involving x_{i-1} and y_i.
        # The constraint for the next boundary i+1 (if i < n-2) is x_{i+1} XOR y_{i+2} = d_{i+1}, involving x_{i+1} and y_{i+2}.
        # The variables {x_i, y_{i+1}} for different values of i (from 0 to n-2) are disjoint.
        # Specifically, the set of variables is {x_0, y_1} for i=0, {x_1, y_2} for i=1, ..., {x_{n-2}, y_{n-1}} for i=n-2.
        # These sets are all disjoint. The variables x_0, ..., x_{n-2} are distinct, and y_1, ..., y_{n-1} are distinct.
        # The variables x_{n-1} and y_0 are not included in any of these constraints.

        # Since the constraints x_i XOR y_{i+1} = d_i are independent for each i, we can minimize the cost for each constraint independently.
        # The total minimum cost will be the sum of minimum costs for each constraint, plus the minimum cost for the unconstrained variables x_{n-1} and y_0.

        # For each i from 0 to n-2:
        # If s[i] == s[i+1] (d_i = 0): We need x_i XOR y_{i+1} = 0, so x_i = y_{i+1}.
        # Options: (x_i, y_{i+1}) = (0, 0) or (1, 1).
        # Cost for (0, 0) is 0*(i+1) + 0*(n-(i+1)) = 0.
        # Cost for (1, 1) is 1*(i+1) + 1*(n-(i+1)) = i+1 + n-i-1 = n.
        # Minimum cost for this boundary is min(0, n) = 0.

        # If s[i] != s[i+1] (d_i = 1): We need x_i XOR y_{i+1} = 1, so x_i != y_{i+1}.
        # Options: (x_i, y_{i+1}) = (0, 1) or (1, 0).
        # Cost for (0, 1) is 0*(i+1) + 1*(n-(i+1)) = n - (i + 1).
        # Cost for (1, 0) is 1*(i+1) + 0*(n-(i+1)) = i + 1.
        # Minimum cost for this boundary is min(i + 1, n - (i + 1)).

        # The minimum cost for the unconstrained variables x_{n-1} and y_0:
        # Cost for x_{n-1} is x_{n-1} * cost(Op1 at n-1) = x_{n-1} * n. Minimum cost is 0 by setting x_{n-1} = 0.
        # Cost for y_0 is y_0 * cost(Op2 at 0) = y_0 * n. Minimum cost is 0 by setting y_0 = 0.

        # The total minimum cost is the sum of the minimum costs for each boundary where s[i] != s[i+1].
        for i in range(n - 1):
            if s[i] != s[i+1]:
                cost_op1_at_i = i + 1
                cost_op2_at_i_plus_1 = n - (i + 1)
                total_cost += min(cost_op1_at_i, cost_op2_at_i_plus_1)

        return total_cost