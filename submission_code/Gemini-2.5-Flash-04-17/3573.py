import collections

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        """
        Counts the number of valid substrings of word1.
        A string x is called valid if x can be rearranged to have word2 as a prefix.
        This is equivalent to checking if the character counts of x are greater than
        or equal to the character counts of word2 for every character ('a' through 'z').
        """
        n = len(word1)
        m = len(word2)

        # Calculate character counts for word2
        count_w2 = [0] * 26
        for char in word2:
            count_w2[ord(char) - ord('a')] += 1

        # Sliding window approach
        current_count_w1 = [0] * 26
        
        # fulfilled_types counts how many of the 26 character types
        # have current_count_w1[c] >= count_w2[c].
        fulfilled_types = 0

        # Initialize fulfilled_types for characters not required by word2 (count_w2[c] == 0)
        # For these characters, the condition current_count_w1[c] >= count_w2[c] (i.e., current_count_w1[c] >= 0)
        # is always true.
        for i in range(26):
            if count_w2[i] == 0:
                fulfilled_types += 1

        total_valid_count = 0
        left = 0 # Left pointer of the sliding window

        for right in range(n):
            # Add the character at the right pointer
            char_right_idx = ord(word1[right]) - ord('a')
            
            # Check if adding this character makes its type fulfilled
            # This happens if the count *was* less than the required count_w2[char_right_idx]
            # and becomes equal to or greater than it after incrementing.
            # We only increment fulfilled_types when the count *exactly* reaches the requirement
            # from below.
            if current_count_w1[char_right_idx] < count_w2[char_right_idx]:
                 current_count_w1[char_right_idx] += 1
                 if current_count_w1[char_right_idx] == count_w2[char_right_idx]:
                    fulfilled_types += 1
            else: # The count was already sufficient (>= count_w2[char_right_idx])
                current_count_w1[char_right_idx] += 1
                # No change to fulfilled_types based on this threshold, as it was already met

            # While the current window word1[left...right] satisfies the condition
            # The condition for validity is: current_count_w1[c] >= count_w2[c] for all 26 characters.
            # This is true if and only if fulfilled_types == 26.
            while fulfilled_types == 26:
                # The current window word1[left...right] is valid.
                # Any substring ending at 'right' with a starting index 'k'
                # from 0 to 'left' is also valid because it contains word1[left...right]
                # and thus has character counts greater than or equal to those in word1[left...right],
                # which are already greater than or equal to those in word2.
                # The number of such valid starting positions for substrings ending at 'right'
                # is (left - 0 + 1) = left + 1.
                total_valid_count += (left + 1)

                # Shrink the window from the left
                char_left_idx = ord(word1[left]) - ord('a')
                
                # Check if removing this character makes its type unfulfilled
                # This happens if the count was exactly equal to count_w2[char_left_idx]
                # before decrementing.
                if current_count_w1[char_left_idx] == count_w2[char_left_idx]:
                    # The count was exactly at the requirement, now drops below after decrement
                    fulfilled_types -= 1
                # Note: If count_w2[char_left_idx] was 0, the condition current_count_w1 >= count_w2 was true.
                # When we decrement current_count_w1 (even from 1 to 0), it remains >= 0.
                # The fulfilled_types logic correctly handles this because it only decrements
                # when current_count_w1 *was* equal to count_w2 (which for count_w2=0 implies count_w1=0)
                # before decrementing, meaning current_count_w1 would become -1, which is impossible.
                # The check `current_count_w1[char_left_idx] == count_w2[char_left_idx]` works
                # correctly even when count_w2[char_left_idx] is 0. If count_w2[idx]=0, we only decrement
                # fulfilled_types if c_w1 was 0 (and becomes -1), which doesn't happen.

                current_count_w1[char_left_idx] -= 1
                left += 1

        return total_valid_count