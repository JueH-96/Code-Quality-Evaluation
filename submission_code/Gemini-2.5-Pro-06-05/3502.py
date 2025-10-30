class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        """
        This method calculates the total number of substrings where at least one
        character appears at least k times.

        The core idea of this O(n^2) approach is as follows:
        1. Iterate through each possible starting index `i` of a substring.
        2. For each `i`, start an inner loop with index `j` from `i` to the
           end of the string. This `j` represents the end of the current substring s[i..j].
        3. Maintain a frequency count of characters for the substring s[i..j].
        4. As we expand the substring by incrementing `j`, we find the *first*
           time a character's frequency reaches `k`.
        5. Once this happens for substring s[i..j], any substring that also starts
           at `i` but ends at an index greater than `j` will also be valid.
        6. The number of such substrings is `n - j`. We add this to our total
           count and break the inner loop to move to the next starting index `i+1`.

        Complexity:
        - Time: O(n^2), where n is the length of the string. The two nested loops
          give a quadratic complexity. For n=3000, n^2 is 9*10^6, which is
          acceptable within typical time limits.
        - Space: O(1), as the frequency array size is constant (26).
        """
        n = len(s)
        total_valid_substrings = 0

        # Iterate through all possible starting positions of a substring
        for i in range(n):
            # For each starting position, expand to the right and track frequencies
            freq = [0] * 26

            for j in range(i, n):
                # Update frequency for the new character s[j]
                char_index = ord(s[j]) - ord('a')
                freq[char_index] += 1

                # If freq[char_index] becomes k, this is the first time any character
                # has reached frequency k for a substring starting at i. If another
                # character had reached k earlier, we would have broken out of this
                # inner loop already.
                if freq[char_index] == k:
                    # All substrings starting at `i` and ending at `j` or any index
                    # after `j` will also be valid.
                    # These substrings are s[i:j+1], s[i:j+2], ..., s[i:n].
                    # The number of such substrings is n - j.
                    total_valid_substrings += (n - j)

                    # Since we have counted all valid substrings starting at `i`,
                    # break from the inner loop and move to the next start index `i+1`.
                    break
        
        return total_valid_substrings