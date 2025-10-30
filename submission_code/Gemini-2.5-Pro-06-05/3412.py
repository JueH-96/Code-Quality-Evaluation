class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        """
        Calculates the permutation difference between two strings s and t.

        The method uses a hash map for efficient O(1) lookups of character indices in t,
        resulting in an overall time complexity of O(N).

        Args:
            s: The first string, with unique characters.
            t: The second string, a permutation of s.

        Returns:
            The permutation difference, which is the sum of the absolute
            differences of character indices between s and t.
        """
        # Create a dictionary to map each character in t to its index for O(1) lookup.
        t_index_map = {char: index for index, char in enumerate(t)}

        # Use a generator expression with sum() to calculate the total difference.
        # For each character and its index `i` in `s`, find its index in `t` from the map,
        # calculate the absolute difference, and sum them up.
        return sum(abs(i - t_index_map[char]) for i, char in enumerate(s))