class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        """
        Calculates the maximum possible achievable number 'x'.

        An integer 'x' is achievable if it can become equal to 'num'
        after applying the following operation no more than 't' times:
        Increase or decrease 'x' by 1, and simultaneously increase or decrease 'num' by 1.

        Let's analyze how the difference (x - num) changes with each operation:
        1. (x+1, num+1): Difference (x+1)-(num+1) = x-num (no change)
        2. (x-1, num-1): Difference (x-1)-(num-1) = x-num (no change)
        3. (x+1, num-1): Difference (x+1)-(num-1) = x-num + 2 (increases by 2)
        4. (x-1, num+1): Difference (x-1)-(num+1) = x-num - 2 (decreases by 2)

        Let x_start be the initial value of x we are looking for.
        Let num_given be the initial value of num.

        We want x_final = num_final after 'k' operations (where k <= t).
        This means the final difference (x_final - num_final) must be 0.

        Let k_minus be the count of operations that decrease the difference by 2 (Type 4).
        Let k_plus be the count of operations that increase the difference by 2 (Type 3).
        Operations of Type 1 and 2 do not change the difference.

        The total change in difference from (x_start - num_given) to (x_final - num_final) is:
        2 * k_plus - 2 * k_minus

        So, (x_final - num_final) = (x_start - num_given) + 2 * k_plus - 2 * k_minus.
        Since we want (x_final - num_final) = 0:
        0 = (x_start - num_given) + 2 * k_plus - 2 * k_minus

        Rearranging to solve for x_start:
        x_start = num_given + 2 * k_minus - 2 * k_plus
        x_start = num_given + 2 * (k_minus - k_plus)

        To maximize x_start, we need to maximize the term (k_minus - k_plus).

        The total number of operations used, k = k_plus + k_minus + k_zero (where k_zero are type 1/2 ops).
        We are given k <= t.
        Since k_zero >= 0, it implies k_plus + k_minus <= t.

        To maximize (k_minus - k_plus) under the constraint k_plus + k_minus <= t:
        We want k_minus to be as large as possible, and k_plus to be as small as possible.
        The smallest possible value for k_plus is 0.
        If k_plus = 0, then the term becomes k_minus.
        The maximum value k_minus can take, given k_minus <= t (since k_plus = 0), is t.

        So, the maximum value for (k_minus - k_plus) is (t - 0) = t.

        Substitute this back into the equation for x_start:
        x_start = num_given + 2 * t

        This means the maximum achievable x is num + 2*t.
        This is achievable by starting with x = num + 2*t and applying 't' operations of type (x-1, num+1).
        Each such operation reduces the difference (x - num) by 2.
        Initial difference = (num + 2*t) - num = 2*t.
        After 't' operations, the difference will be 2*t - 2*t = 0, meaning x = num.
        Total operations used = t, which satisfies the condition.
        """
        return num + 2 * t