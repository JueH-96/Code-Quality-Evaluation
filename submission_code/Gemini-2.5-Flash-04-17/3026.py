class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        """
        Calculates the minimum possible sum of a beautiful array of length n.

        Args:
            n: The length of the array.
            target: The target sum that should not be formed by two distinct elements.

        Returns:
            The minimum possible sum modulo 10^9 + 7.
        """
        MOD = 10**9 + 7

        # The numbers 1, 2, ..., target // 2 can be picked together
        # without any pair summing to target.
        # Let count_initial_pick = target // 2.
        # Any two distinct numbers i, j from {1, ..., target // 2} sum to i+j <= (target//2 - 1) + target//2.
        # If target is even, target = 2m, target//2 = m. Max sum is (m-1) + m = 2m-1 < target.
        # If target is odd, target = 2m+1, target//2 = m. Max sum is m + m = 2m < target.
        # So the set {1, 2, ..., target // 2} is always beautiful.

        count_initial_pick = target // 2

        if n <= count_initial_pick:
            # We only need n numbers and can pick the smallest n positive integers: 1, 2, ..., n.
            # Since n <= count_initial_pick = target // 2, the set {1, ..., n} is beautiful.
            # The minimum sum is the sum of 1 to n.
            # Sum of 1 to n = n * (n + 1) // 2

            # Use Python's arbitrary precision integers. Perform // 2 first, then % MOD.
            ans = (n * (n + 1) // 2) % MOD

        else:
            # We pick the first count_initial_pick numbers: 1, 2, ..., count_initial_pick.
            # Sum of initial pick = count_initial_pick * (count_initial_pick + 1) // 2

            sum_initial = (count_initial_pick * (count_initial_pick + 1) // 2) % MOD

            # We need n - count_initial_pick more numbers.
            n_rem = n - count_initial_pick

            # The numbers picked are {1, ..., count_initial_pick}.
            # A number j is available if j > count_initial_pick and target - j is not in {1, ..., count_initial_pick}.
            # target - j not in {1, ..., count_initial_pick} is equivalent to (target - j < 1) or (target - j > count_initial_pick).
            # target - j < 1  => j > target - 1.
            # target - j > count_initial_pick => j < target - count_initial_pick.
            # We are looking for j > count_initial_pick.
            # So, j > count_initial_pick AND (j > target - 1 OR j < target - count_initial_pick).
            # Since count_initial_pick = target // 2, target - count_initial_pick >= target - target/2 = target/2 = count_initial_pick.
            # Thus, j < target - count_initial_pick is only possible if target - count_initial_pick > count_initial_pick,
            # i.e., target > 2 * count_initial_pick. This is true if target is odd.
            # If target is odd, target = 2m+1, count_initial_pick = m. j < (2m+1) - m = m+1.
            # So j < m+1. Combined with j > count_initial_pick = m, this gives m < j < m+1, which has no integer solution.
            # If target is even, target = 2m, count_initial_pick = m. j < 2m - m = m.
            # So j < m. Combined with j > count_initial_pick = m, this gives m < j < m, which has no integer solution.
            # In both cases, there are no integers j such that count_initial_pick < j < target - count_initial_pick.
            # So the condition (j > count_initial_pick AND j < target - count_initial_pick) is never met for j > count_initial_pick.
            # The only way for j > count_initial_pick to be available is j > target - 1, i.e., j >= target.
            # The smallest available integers after 1..count_initial_pick are target, target + 1, target + 2, ...

            # We need n_rem numbers from this sequence.
            # The numbers are target, target + 1, ..., target + n_rem - 1.
            # Sum of arithmetic series: m * (first + last) // 2
            # m = n_rem
            # first = target
            # last = target + n_rem - 1
            # sum_rem = n_rem * (target + target + n_rem - 1) // 2
            # sum_rem = n_rem * (2 * target + n_rem - 1) // 2

            sum_rem = (n_rem * (2 * target + n_rem - 1) // 2) % MOD

            ans = (sum_initial + sum_rem) % MOD

        return ans