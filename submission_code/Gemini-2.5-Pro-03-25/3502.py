import collections # Import not needed for this implementation, but generally useful

class Solution:
    """
    Calculates the total number of substrings of s where at least one character appears at least k times.

    The approach uses the principle of inclusion-exclusion by calculating the complement:
    1. Calculate the total number of possible substrings in the string `s`. The formula for this is n * (n + 1) / 2, where n is the length of `s`.
    2. Calculate the number of "bad" substrings. A "bad" substring is defined as one where *all* characters appear strictly less than k times.
    3. The number of "good" substrings (where at least one character appears at least k times) is then the total number of substrings minus the number of "bad" substrings.

    Step 2 (calculating the count of "bad" substrings) is efficiently done using a sliding window approach, which takes O(n) time complexity:
    - We maintain a sliding window defined by `left` and `right` pointers, representing the substring `s[left:right+1]`.
    - We also maintain a frequency map (`freq`) of characters within the current window.
    - We iterate through the string with the `right` pointer from 0 to n-1, expanding the window to the right.
    - For each position of `right`, we update the frequency count for the character `s[right]`.
    - After expanding, we check if the current window `s[left:right+1]` violates the "bad" substring condition (i.e., if it contains any character with a frequency of `k` or more).
    - If the window violates the condition, we shrink it from the `left` side by incrementing `left` and decrementing the frequency count for `s[left]`. We repeat this shrinking process until the window `s[left:right+1]` satisfies the "bad" substring condition (all character frequencies are strictly less than `k`).
    - Once the window `s[left:right+1]` satisfies the "bad" condition, we know that all substrings ending at the current `right` index and starting at an index `i` such that `left <= i <= right` are "bad" substrings. The number of such substrings is `right - left + 1`. We add this number to our running count of "bad" substrings (`count_less_k`).
    - By the end of the iteration, `count_less_k` will hold the total number of "bad" substrings in `s`.
    - The final result is `total_substrings - count_less_k`.

    This approach ensures that both the `left` and `right` pointers only move forward through the string, leading to an overall time complexity of O(n) because the work inside the loops (frequency updates and checks) takes constant time relative to n (O(26) for the frequency check). The space complexity is O(1) as the frequency map has a fixed size (26 for lowercase English letters).
    """
    def numberOfSubstrings(self, s: str, k: int) -> int:
        """
        Args:
          s: The input string, consisting of lowercase English letters.
          k: The minimum frequency threshold for at least one character in a valid substring.

        Returns:
          The total number of substrings of s where at least one character appears at least k times.
        """
        n = len(s)

        # 1. Calculate the total number of possible substrings.
        # A string of length n has n*(n+1)/2 substrings.
        total_substrings = n * (n + 1) // 2

        # Handle the edge case where k=1. If k is 1, any non-empty substring
        # will have at least one character appearing at least once.
        # Therefore, all n*(n+1)/2 substrings are valid.
        if k == 1:
            return total_substrings

        # 2. Calculate the number of "bad" substrings using a sliding window.
        # A "bad" substring is one where all character frequencies are strictly less than k.
        count_less_k = 0 # Accumulates the count of "bad" substrings.
        left = 0         # Left boundary of the sliding window.
        # Frequency map for characters in the window s[left:right+1].
        # Using a list of size 26 for O(1) access based on character ASCII values.
        freq = [0] * 26

        # Iterate through the string with the right pointer to expand the window.
        for right in range(n):
            # Expand the window by including the character s[right].
            char_right_index = ord(s[right]) - ord('a')
            freq[char_right_index] += 1

            # Check if the current window s[left:right+1] violates the "bad" condition
            # (i.e., if it contains at least one character with frequency >= k).
            # If it does, shrink the window from the left until it becomes "bad" again.
            while True:
                # Determine if there's a violation (any frequency >= k).
                violation = False
                # This check takes O(26) time, which is constant.
                for i in range(26):
                     if freq[i] >= k:
                         violation = True
                         break

                if not violation:
                    # If no frequency is >= k, the window s[left:right+1] is "bad"
                    # (satisfies the condition: all frequencies < k). Stop shrinking.
                    break
                else:
                    # If there is a violation, we must shrink the window from the left.
                    # Decrement the frequency of the character at the left pointer.
                    char_left_index = ord(s[left]) - ord('a')
                    freq[char_left_index] -= 1
                    # Move the left pointer one step to the right.
                    left += 1
                    # The loop continues to check the new, smaller window.

            # After the shrinking phase (if any), the window s[left:right+1]
            # is guaranteed to have all character frequencies < k.
            # Therefore, any substring ending at `right` and starting at an index `i`
            # such that `left <= i <= right` is a "bad" substring.
            # The number of such substrings is `right - left + 1`.
            # Add this count to the total count of "bad" substrings found so far.
            count_less_k += (right - left + 1)

        # 3. The final result is the total number of substrings minus the count of "bad" ones.
        return total_substrings - count_less_k