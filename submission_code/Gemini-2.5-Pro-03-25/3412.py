import collections # This import is not strictly necessary if using dict comprehension

class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        """
        Calculates the permutation difference between two strings s and t.

        The permutation difference is defined as the sum of the absolute differences
        between the index of each character in s and its index in t.

        Args:
            s: The first string. Each character occurs at most once.
            t: The second string, which is a permutation of s.

        Returns:
            The permutation difference as an integer.

        Constraints:
            1 <= s.length <= 26
            Each character occurs at most once in s.
            t is a permutation of s.
            s consists only of lowercase English letters.

        Example 1:
            Input: s = "abc", t = "bac"
            Output: 2
            Explanation: |index_s('a') - index_t('a')| + |index_s('b') - index_t('b')| + |index_s('c') - index_t('c')|
                       = |0 - 1| + |1 - 0| + |2 - 2| = 1 + 1 + 0 = 2

        Example 2:
            Input: s = "abcde", t = "edbac"
            Output: 12
            Explanation: |0-3| + |1-2| + |2-4| + |3-1| + |4-0| = 3 + 1 + 2 + 2 + 4 = 12

        Approach:
            1. Create a hash map (dictionary in Python) to store the index of each character in string t. This allows for efficient O(1) average time lookup.
            2. Iterate through string s, keeping track of the current character and its index.
            3. For each character in s, look up its index in t using the hash map.
            4. Calculate the absolute difference between the index in s and the index in t.
            5. Sum up these absolute differences.
            6. Return the total sum.

        Time Complexity: O(N), where N is the length of the strings (N <= 26).
                         - O(N) to build the index map for t.
                         - O(N) to iterate through s and calculate differences.
                         - Overall O(N).
        Space Complexity: O(N) for the dictionary storing indices of t. Since N <= 26 and the
                          character set is fixed (lowercase English letters), this can also be
                          considered O(1) constant space.
        """

        # Step 1: Create a map to store the index of each character in string t.
        # Key: character, Value: index in t
        t_index_map = {char: index for index, char in enumerate(t)}
        # Example: if t = "bac", t_index_map = {'b': 0, 'a': 1, 'c': 2}

        permutation_difference = 0
        # Step 2 & 3: Iterate through string s with index.
        for index_s, char_s in enumerate(s):
            # Step 3a: Find the index of the same character in string t using the map.
            # Since t is a permutation of s and characters in s are unique,
            # char_s is guaranteed to be a key in t_index_map.
            index_t = t_index_map[char_s]

            # Step 4: Calculate the absolute difference of indices.
            difference = abs(index_s - index_t)

            # Step 5: Add the difference to the total permutation difference.
            permutation_difference += difference

        # Step 6: Return the total computed difference.
        return permutation_difference