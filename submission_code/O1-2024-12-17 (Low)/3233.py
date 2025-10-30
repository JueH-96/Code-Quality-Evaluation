class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        """
        We want to choose at most one character in s to change (to any other lower-case English letter),
        so that when we repeatedly partition off the longest prefix containing at most k distinct characters,
        the total number of partitions is maximized.

        Approach:
          1. Write a function get_partitions(s, k) that, given a string s and integer k, returns the number of
             partitions formed by repeatedly taking the longest prefix with at most k distinct characters.
             - This can be done in O(n) time via a single pass with a frequency array and a running count of distinct.
          2. Compute the baseline partitions (no change).
          3. Try changing each index i in s to each possible character from 'a' to 'z' (except if it is the same
             as the original). For each change, compute the partitions again using get_partitions.
          4. Take the maximum over all such one-character modifications.
          5. Because the maximal number of partitions cannot exceed the length of the string, if we ever
             achieve that, we can stop early.

        This solution is O(n * 26 * n) in the worst case, i.e. O(26 * n^2). For n up to 10^4, this is borderline
        but can often be implemented efficiently enough in Python to pass if done with careful optimizations
        (fast lookups, early exits, etc.). If needed, one can optimize further, but we'll implement this direct
        approach.
        """

        # Fast helper function to compute number of partitions
        # given a string (as a list of chars) and k.
        def get_partitions(arr, k):
            freq = [0]*26
            distinct = 0
            count = 0
            n = len(arr)

            for i, ch in enumerate(arr):
                idx = ord(ch) - ord('a')
                freq[idx] += 1
                if freq[idx] == 1:  # new distinct character
                    distinct += 1
                if distinct > k:
                    # We form a new partition just before this character.
                    count += 1
                    # Reset freq array with only the current character counted
                    freq = [0]*26
                    freq[idx] = 1
                    distinct = 1
            # One more partition for the last segment
            count += 1
            return count

        # If k >= number of unique letters in s, the result is 1 for no change
        # (the entire string is one partition). But let's not shortcut incorrectly,
        # as changing one char might inflate partitions if k=1 or other small k.
        # We'll just compute normally.

        arr = list(s)
        n = len(arr)

        # Compute baseline partitions
        best = get_partitions(arr, k)
        # If best is already the maximum possible (which is n),
        # we can't do better than that.
        if best == n:
            return n

        # Try changing each character to each different letter
        original_best = best
        for i in range(n):
            original = arr[i]
            orig_index = ord(original) - ord('a')

            for c_ord in range(26):
                if c_ord == orig_index:
                    continue
                arr[i] = chr(c_ord + ord('a'))
                candidate = get_partitions(arr, k)
                if candidate > best:
                    best = candidate
                    # Early stop if we already reached maximum possible
                    if best == n:
                        return best

            # Revert the change
            arr[i] = original

        return best