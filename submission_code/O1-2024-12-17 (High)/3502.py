class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # Total number of substrings
        total_substrings = n * (n + 1) // 2
        
        # We will count how many substrings ("bad") do NOT have any character appearing >= k times
        # Then we subtract that from total_substrings to get the answer.
        
        freq = [0] * 26  # frequency array for letters 'a'..'z'
        start = 0
        max_freq = 0
        bad_count = 0
        
        for end in range(n):
            # Expand the window to include s[end]
            right_char_index = ord(s[end]) - ord('a')
            freq[right_char_index] += 1
            
            # Update the maximum frequency in the current window
            if freq[right_char_index] > max_freq:
                max_freq = freq[right_char_index]
            
            # Shrink from the left while we have a character with frequency >= k
            while max_freq >= k and start <= end:
                left_char_index = ord(s[start]) - ord('a')
                freq[left_char_index] -= 1
                start += 1
                # Recompute max frequency (constant time since it's only 26 letters)
                max_freq = max(freq)
            
            # Now the window [start..end] is "bad" (no character repeated k times),
            # so all subwindows that end at 'end' and start in [start..end] are also "bad".
            bad_count += (end - start + 1)
        
        # Substrings that have at least one character repeated k times = total - bad
        return total_substrings - bad_count