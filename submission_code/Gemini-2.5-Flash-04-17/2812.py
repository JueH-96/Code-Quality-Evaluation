class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # Let the initial number we are looking for be x. The initial given number is num.
        # The operation is: increase or decrease x by 1, and simultaneously increase or decrease num by 1.
        # Let the initial values be x_0 and num_0 (where num_0 is the input 'num').
        # Let the values after k operations be x_k and num_k.
        # We want x_k = num_k for some k <= t.

        # Let's analyze the change in the difference (x - num) after one operation.
        # Let dx be the change in x, and dn be the change in num. |dx|=1, |dn|=1.
        # Change in (x - num) = (x_0 + dx) - (num_0 + dn) - (x_0 - num_0) = dx - dn.
        # Possible values for (dx, dn): (+1, +1), (+1, -1), (-1, +1), (-1, -1).
        # Possible values for (dx - dn):
        # (+1) - (+1) = 0
        # (+1) - (-1) = 2
        # (-1) - (+1) = -2
        # (-1) - (-1) = 0
        # Each operation changes the difference (x - num) by either +2, -2, or 0.

        # Let D_0 = x_0 - num_0 be the initial difference.
        # After k operations, the final difference D_k = x_k - num_k.
        # The change in the difference over k steps is the sum of changes in each step.
        # Total change in (x - num) = sum(dx_i - dn_i) for i=1 to k.
        # Since each (dx_i - dn_i) is in {0, 2, -2}, the sum is always an even number.
        # D_k = D_0 + sum(dx_i - dn_i).
        # We want D_k = 0. So, D_0 = -sum(dx_i - dn_i) = sum(dn_i - dx_i).

        # Let k_p2 be the number of times (dn_i - dx_i) = +2. This happens when dn=+1, dx=-1 (decrease x, increase num).
        # Let k_m2 be the number of times (dn_i - dx_i) = -2. This happens when dn=-1, dx=+1 (increase x, decrease num).
        # Let k_z be the number of times (dn_i - dx_i) = 0. This happens for the other two operations.
        # The total change sum(dn_i - dx_i) = k_p2 * (+2) + k_m2 * (-2) + k_z * (0) = 2 * (k_p2 - k_m2).

        # So, the initial difference D_0 = x_0 - num_0 = 2 * (k_p2 - k_m2).
        # The total number of operations k = k_p2 + k_m2 + k_z <= t.
        # We want to maximize x_0, which means maximizing D_0 = x_0 - num_0.
        # This implies maximizing (k_p2 - k_m2).
        # To maximize (k_p2 - k_m2) subject to k_p2 + k_m2 + k_z <= t and k_p2, k_m2, k_z >= 0:
        # Maximize k_p2 and minimize k_m2.
        # The maximum value of k_p2 is t (when k_m2 = 0, k_z = 0, using exactly t operations).
        # The minimum value of k_m2 is 0.
        # The maximum value of (k_p2 - k_m2) is t - 0 = t.

        # This maximum difference is achieved when k_p2 = t, k_m2 = 0, k_z = 0. This means performing
        # the operation "decrease x by 1, increase num by 1" exactly t times.
        # The initial difference x_0 - num_0 must be 2 * t.
        # So, the maximum x_0 is num_0 + 2 * t.

        # In the function, num_0 is the input 'num'. The maximum achievable x is num + 2*t.
        return num + 2 * t