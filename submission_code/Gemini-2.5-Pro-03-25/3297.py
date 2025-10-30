import math # Not needed, integer arithmetic is used

class Solution:
    """
    This class provides a solution to find the minimum time to return 
    a string to its initial state after applying specific operations described
    in the problem statement.
    """
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        """
        Calculates the minimum time greater than zero required for the word 
        to revert to its initial state.

        At every second, the following operations are performed:
        1. Remove the first k characters of word.
        2. Add any k characters to the end of word.
        The characters added do not need to be the same as those removed.

        Args:
            word: The initial 0-indexed string.
            k: An integer representing the number of characters to remove and add 
               at each step.

        Returns:
            The minimum time (number of seconds) greater than zero required for the word
            to potentially revert to its initial state `word`.
        """
        n = len(word)
        
        # Calculate ceil(n / k) using integer arithmetic. Let this be t_max.
        # t_max represents the minimal time by which the first n characters (the entire original word)
        # would have been removed from the prefix if we kept track of original positions.
        # It serves as an upper bound for the minimum time required. Why? Because at time t_max, 
        # the resulting word consists entirely of characters derived from the added suffixes S_1, ..., S_t_max.
        # We have full control over these S_i, so we can choose them strategically such that the final word
        # becomes exactly the original `word`. Therefore, restoration is always possible by time t_max.
        t_max = (n + k - 1) // k

        # We are looking for the *minimum* time t > 0.
        # Let's check times t starting from 1 up to t_max - 1.
        # Restoration at time t is possible if and only if the structure of the original `word` allows it.
        # The condition for this possibility is that the suffix of `word` starting at index `t*k`
        # must be identical to the prefix of `word` of length `n - t*k`.
        # If `word[t*k : n] == word[0 : n - t*k]`, then we can choose the added suffixes S_1, ..., S_t
        # such that the word state at time t becomes equal to the original `word`.
        
        for t in range(1, t_max):
            # Python slicing automatically handles boundaries.
            # word[t*k:] extracts the substring from index t*k to the end.
            # word[:n - t*k] extracts the substring from index 0 up to (but not including) index n - t*k.
            
            # Check if the suffix matches the prefix.
            # For t in range(1, t_max), we are guaranteed that t*k < n, 
            # ensuring that this comparison is meaningful (not comparing empty strings due to indices out of bounds).
            if word[t * k:] == word[:n - t * k]:
                 # If a match is found, this value of t is the minimum possible time > 0,
                 # because we are checking t in increasing order (1, 2, ...).
                return t
                
        # If the loop completes without finding any t < t_max that satisfies the condition,
        # it implies that restoration is not possible before time t_max.
        # Since restoration is guaranteed to be possible at time t_max,
        # the minimum time required must be t_max.
        return t_max