import math

class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # At each operation we remove the first k characters
        # and append k arbitrary characters. This means that after t operations,
        # the first n - t*k characters of the current word are forced:
        # they come directly from the original word (shifted t * k positions).
        # In other words, after t seconds when t*k < n, the word becomes:
        #   forced_part = word[t*k:]   (of length n - t*k)
        #   free_part   = (t*k positions, chosen arbitrarily)
        #
        # To have the final word equal to the initial word, we must have:
        #    forced_part == word[:n - t*k]
        # because we may choose the free part arbitrarily to fill
        # the remainder.
        #
        # Also, if t*k >= n then the entire word becomes free
        # (because removed forced part is of length ≤ 0)
        # and we can simply choose all characters to match word.
        # Therefore, one valid answer is always t = ceil(n/k) (if no earlier t works).
        
        # Let t vary over operations that leave a forced part (i.e. t*k < n)
        max_t = math.ceil(n / k)
        for t in range(1, max_t):
            # forced part length = n - t*k (this is positive by loop constraint)
            forced = word[t*k:]
            prefix = word[:n - t*k]
            if forced == prefix:
                return t
        return max_t

# For simple testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.minimumTimeToInitialState("abacaba", 3))  # Expected output: 2
    # Example 2:
    print(sol.minimumTimeToInitialState("abacaba", 4))  # Expected output: 1
    # Example 3:
    print(sol.minimumTimeToInitialState("abcbabcd", 2)) # Expected output: 4