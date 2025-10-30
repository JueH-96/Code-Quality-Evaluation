class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        """
        Calculates the maximum achievable number x.

        An integer x is achievable if it can become equal to num after applying
        the following operation no more than t times:
        Increase or decrease x by 1, and simultaneously increase or decrease num by 1.

        To maximize the initial value of x, we should use the `t` operations to
        bridge the largest possible gap between x and num. The most effective
        operation for reducing a gap where x > num is to decrease x by 1 and
        increase num by 1. This reduces their difference by 2.

        If we apply this operation `t` times, the new values will be:
        - new_x = x - t
        - new_num = num + t

        For these to be equal, we set `x - t = num + t`, which gives `x = num + 2*t`.
        This shows that `num + 2*t` is an achievable number. Any larger `x` would
        require more than `t` operations to make it equal to `num`.
        """
        
        # Each operation can increase the final value of `num` by at most 1, so after `t` operations,
        # the maximum possible value for `num` is `num + t`.
        # To reach this value, `x` must be decreased. Each operation can decrease `x` by at most 1.
        # After `t` operations, the minimum value for `x` is `x - t`.
        # To make them equal: x - t = num + t  =>  x = num + 2 * t
        
        return num + 2 * t