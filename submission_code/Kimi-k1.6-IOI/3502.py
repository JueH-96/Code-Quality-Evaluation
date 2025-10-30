class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = n * (n + 1) // 2
        
        # Calculate the number of invalid substrings (all characters < k)
        count_invalid = 0
        freq = [0] * 26
        left = 0
        counter = 0  # number of characters in the window with frequency >= k
        
        for right in range(n):
            # Add the current character to the window
            idx = ord(s[right]) - ord('a')
            old = freq[idx]
            freq[idx] += 1
            new = freq[idx]
            
            # Update the counter based on the change in frequency
            delta = (new >= k) - (old >= k)
            counter += delta
            
            # Shrink the window from the left until all frequencies are < k
            while counter > 0:
                left_char = s[left]
                left_idx = ord(left_char) - ord('a')
                old_left = freq[left_idx]
                freq[left_idx] -= 1
                new_left = freq[left_idx]
                
                delta_left = (new_left >= k) - (old_left >= k)
                counter += delta_left
                left += 1
            
            # All characters in the current window [left, right] have frequency < k
            count_invalid += (right - left + 1)
        
        return total - count_invalid