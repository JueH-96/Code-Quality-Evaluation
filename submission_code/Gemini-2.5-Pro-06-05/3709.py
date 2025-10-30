import itertools

class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        """
        Determines if there exists a special substring of length k in s.

        A special substring of length k has three properties:
        1. It has a length of exactly k.
        2. It consists of only one distinct character (e.g., "aaa").
        3. It is "isolated": the characters immediately before and after it
           (if they exist) are different from the character in the substring.

        Args:
            s: The input string.
            k: The required length of the substring.

        Returns:
            True if such a substring exists, False otherwise.
        """
        
        # The problem is equivalent to finding a run of a single character
        # that has a length of exactly k. A "run" is a maximal contiguous
        # sequence of identical characters. By its nature, a run is preceded
        # and succeeded by different characters or the boundaries of the string.

        # itertools.groupby is the ideal tool for this task. It iterates
        # over the string and groups consecutive identical characters.
        for _, group in itertools.groupby(s):
            # The 'group' is an iterator over the characters in the current run.
            # To find the length of the run, we can count the elements in the iterator.
            # A simple way is to convert the iterator to a list and get its length.
            run_length = len(list(group))
            
            # If we find a run whose length is exactly k, we have found
            # our special substring.
            if run_length == k:
                return True
        
        # If we have iterated through all the runs and haven't found one
        # of length k, then no such special substring exists.
        return False