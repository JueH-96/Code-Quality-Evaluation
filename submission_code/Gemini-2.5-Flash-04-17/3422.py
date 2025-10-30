class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        """
        Calculates the value of a[n-1] after k seconds modulo 10^9 + 7.

        Args:
            n: The size of the array.
            k: The number of seconds.

        Returns:
            The value of a[n-1] after k seconds modulo 10^9 + 7.
        """
        MOD = 10**9 + 7

        # Initialize the array a with all ones.
        # This array will store the state of the array after each second.
        # a[i] represents the value at index i.
        a = [1] * n

        # Simulate the process for k seconds.
        # Each second, the array is updated according to the rule.
        for _ in range(k):
            # After each second, update each element a[i]
            # to be the sum of all its preceding elements plus the element itself.
            # This operation is a prefix sum transformation on the array from the previous second.
            # Let a_old be the array state before the update (at second s-1).
            # The new state a_new (at second s) is given by a_new[i] = sum(a_old[0]...a_old[i]).
            # This can be computed iteratively: a_new[0] = a_old[0], and a_new[i] = a_new[i-1] + a_old[i] for i > 0.
            # We can perform this update efficiently in place by iterating from left to right.
            # When we are at index `i` (starting from 1), `a[i-1]` already holds the
            # newly computed value for `a_new[i-1]`, while `a[i]` still holds the
            # old value from the previous second `a_old[i]`.
            # Thus, setting a[i] = (a[i-1] + a[i]) % MOD correctly computes a_new[i].
            
            # The first element a[0] implicitly remains the same (sum of first element is just the element itself).
            # We only need to update elements from index 1 onwards.
            for i in range(1, n):
                # Update a[i] to the sum of elements up to index i from the previous second.
                # This is achieved by adding the cumulative sum up to i-1 (which is now stored in a[i-1])
                # and the element at i from the previous second (which is still stored in a[i]).
                a[i] = (a[i] + a[i-1]) % MOD

        # After k seconds, the array 'a' contains the final state.
        # The value of a[n-1] is the required answer.
        return a[n - 1]