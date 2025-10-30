class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)

        # 1. Calculate the frequency map of required characters from word2.
        freq2 = [0] * 26
        for char in word2:
            freq2[ord(char) - ord('a')] += 1

        # 2. Initialize variables for the two-pointer sliding window.
        # `deficient_count` tracks how many character requirements are unmet.
        # The window is valid when `deficient_count` becomes 0.
        deficient_count = 0
        for count in freq2:
            if count > 0:
                deficient_count += 1

        total_valid_count = 0
        # `current_freq` stores the character counts for the current window [i, j-1].
        current_freq = [0] * 26
        # `j` is the right pointer of the window, exclusive.
        j = 0

        # 3. Iterate with the left pointer `i` to define the start of substrings.
        for i in range(n):
            # When `i` moves, `word1[i-1]` is removed from the window's left side.
            if i > 0:
                char_code_rem = ord(word1[i-1]) - ord('a')
                # If the removed character's count was exactly meeting its requirement,
                # the requirement now becomes unmet (deficient).
                if current_freq[char_code_rem] == freq2[char_code_rem]:
                    deficient_count += 1
                current_freq[char_code_rem] -= 1

            # Expand the window to the right with `j` until all requirements are met.
            while j < n and deficient_count > 0:
                char_code_add = ord(word1[j]) - ord('a')
                current_freq[char_code_add] += 1
                # If adding this character satisfies its requirement, one deficiency is resolved.
                if current_freq[char_code_add] == freq2[char_code_add]:
                    deficient_count -= 1
                j += 1
            
            # If all requirements are met, the window `word1[i:j]` is the shortest
            # valid substring starting at `i`. Any substring starting at `i` and
            # ending at or after `j-1` is also valid.
            if deficient_count == 0:
                # Valid substrings are `word1[i:j], word1[i:j+1], ..., word1[i:n]`.
                # The number of such substrings is `n - j + 1`.
                total_valid_count += (n - j + 1)
        
        return total_valid_count