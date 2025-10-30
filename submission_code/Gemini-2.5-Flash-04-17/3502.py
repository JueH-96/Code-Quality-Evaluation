import string

class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        
        # Total number of substrings in s
        total_substrings = n * (n + 1) // 2
        
        # Count substrings where all character frequencies are strictly less than k.
        # This is equivalent to counting substrings where all character frequencies are at most k-1.
        # We use a helper function `count_at_most_freq(s, threshold)` for this.
        bad_substrings = self.count_at_most_freq(s, k - 1)
        
        # The number of substrings where at least one character appears at least k times
        # is the total number of substrings minus the number of substrings where
        # all characters appear less than k times.
        return total_substrings - bad_substrings

    def count_at_most_freq(self, s: str, threshold: int) -> int:
        """
        Counts the number of substrings in s where all character frequencies
        are at most the given threshold.
        Uses a two-pointer sliding window approach.
        """
        n = len(s)
        left = 0
        count = 0
        # Use a dictionary for frequency counting
        freq = {char: 0 for char in string.ascii_lowercase}
        
        # This counter tracks how many distinct characters have frequency strictly greater than threshold
        # in the current window s[left:right+1].
        # When this counter is 0, all characters in the current window s[left:right+1]
        # have frequency <= threshold, meaning the window is valid according to the condition.
        num_violations = 0 

        for right in range(n):
            char_right = s[right]
            
            # Update frequency for the character entering the window
            freq[char_right] += 1
            
            # Check if this character's frequency now violates the threshold (becomes > threshold)
            # This happens when the frequency transitions from threshold to threshold + 1.
            if freq[char_right] == threshold + 1:
                num_violations += 1
                
            # While there are any characters with frequency > threshold in the current window
            # s[left:right+1] (i.e., num_violations > 0), the window is invalid according to the condition.
            # We must shrink the window from the left until it becomes valid again.
            while num_violations > 0:
                char_left = s[left]
                
                # Update frequency for the character leaving the window
                freq[char_left] -= 1
                
                # Check if this character's frequency now stops violating the threshold
                # (transitions from threshold + 1 back down to threshold).
                # If it does, we decrement the violation counter.
                if freq[char_left] == threshold:
                    num_violations -= 1
                    
                # Move the left pointer to shrink the window
                left += 1
            
            # After the while loop, the window s[left:right+1] is valid (all character frequencies <= threshold).
            # For a fixed right endpoint, any substring s[left':right+1] where left <= left' <= right
            # is a sub-window of s[left:right+1]. Character counts in a sub-window are less than or equal
            # to the counts in the larger window. Therefore, if all counts in s[left:right+1] are <= threshold,
            # all counts in any sub-window s[left':right+1] must also be <= threshold.
            # This means all substrings ending at 'right' with start index from 'left' to 'right'
            # satisfy the condition.
            # These substrings are s[left:right+1], s[left+1:right+1], ..., s[right:right+1].
            # The number of such substrings is (right - left + 1).
            # We add this count to our total count of valid substrings ending at 'right'.
            count += (right - left + 1)
            
        # The function returns the total count of substrings satisfying the "at most threshold frequency" condition.
        return count