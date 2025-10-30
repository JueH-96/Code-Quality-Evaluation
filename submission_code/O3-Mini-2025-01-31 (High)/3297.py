import math

class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # Think of the process as a queue.
        # At every move we remove the first k characters
        # and then append k characters (which we can choose freely).
        # However, the characters that “survive” (i.e. that were never removed)
        # appear in the final word in the same order as in the original word.
        #
        # Suppose we perform T moves and T*k < n.
        # Then exactly n - T*k characters (the ones at the very end) 
        # of the original word still remain in the final string, in order.
        # Denote this forced block by: forced = word[T*k : n] (its length is n - T*k).
        #
        # In order to reconstruct the initial word, we must fill
        # the remaining T*k positions (by choosing the appended characters appropriately)
        # so that the final word equals the original.
        # In other words, the forced part must already match the prefix of the original word.
        # That is, we need
        #      word[0 : n - T*k] == word[T*k : n]
        # If that happens for some T (with T*k < n), then we can choose the appended blocks
        # to “complete” the word.
        #
        # Otherwise, if no such T exists (or T*k >= n),
        # then at some point we have removed all original characters.
        # Since we may choose the appended characters arbitrarily,
        # one valid strategy is to simply overwrite the entire word.
        # The minimum number of moves required for that is T = ceil(n / k).
        #
        # So we try for T = 1,2,... as long as T*k < n:
        max_candidate = (n - 1) // k  # maximum T such that T*k < n
        for T in range(1, max_candidate + 1):
            border_len = n - T * k
            # The surviving block is forced to be what was originally at word[T*k:n].
            # For the final word to equal the original word,
            # it is necessary that word[T*k:n] equals the prefix word[0:border_len].
            if word[:border_len] == word[T*k:]:
                return T
        
        # If no valid T is found among the ones with T*k < n,
        # then we can overwrite everything by doing T = ceil(n/k) moves.
        return math.ceil(n / k)


# Example usage and basic tests:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    # "abacaba", k = 3. We must check if a move less than full replacement works.
    # For T = 1, forced part is word[3:7] = "caba" vs word[0:4] = "abac" (mismatch).
    # For T = 2, forced part is word[6:7] = "a" vs word[0:1] = "a" (match).
    # Thus answer is 2.
    print(sol.minimumTimeToInitialState("abacaba", 3))  # Expected: 2

    # Example 2:
    # "abacaba", k = 4. For T = 1, forced part is word[4:7] = "aba" vs word[0:3] = "aba".
    # So answer is 1.
    print(sol.minimumTimeToInitialState("abacaba", 4))  # Expected: 1

    # Example 3:
    # "abcbabcd", k = 2.
    # Try T = 1: forced part length = 8 - 2 = 6, compare word[0:6] vs word[2:8]: "abcbab" vs "cbabcd" (no)
    # T = 2: forced part length = 8 - 4 = 4, compare word[0:4] vs word[4:8]: "abcb" vs "abcd" (no)
    # T = 3: forced part length = 8 - 6 = 2, compare word[0:2] vs word[6:8]: "ab" vs "cd" (no)
    # Since no such T exists with surviving characters,
    # we return ceil(8/2) = 4.
    print(sol.minimumTimeToInitialState("abcbabcd", 2))  # Expected: 4