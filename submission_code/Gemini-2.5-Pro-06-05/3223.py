class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        
        # Helper function to count complete substrings within a "valid segment".
        # A valid segment is a part of the word where adjacent characters differ by at most 2.
        def _count_in_segment(segment: str) -> int:
            segment_len = len(segment)
            count = 0
            
            # Iterate through all possible numbers of unique characters (u).
            # A complete substring with `u` unique characters must have length u * k.
            for u in range(1, 27):
                window_len = u * k
                if window_len > segment_len:
                    break
                
                freq = [0] * 26
                unique_chars = 0  # Count of unique characters in the current window
                chars_at_k = 0    # Count of characters with frequency exactly k
                
                # Sliding window of size `window_len` over the segment
                for i in range(segment_len):
                    # --- Add character at index i to the window ---
                    idx_add = ord(segment[i]) - ord('a')
                    old_freq_add = freq[idx_add]
                    freq[idx_add] += 1
                    
                    # Update counters based on the added character
                    if old_freq_add == 0:
                        unique_chars += 1
                    if old_freq_add == k - 1:
                        chars_at_k += 1
                    elif old_freq_add == k:
                        chars_at_k -= 1
                    
                    # --- Remove character from the left of the window if it's full ---
                    if i >= window_len:
                        idx_rem = ord(segment[i - window_len]) - ord('a')
                        old_freq_rem = freq[idx_rem]
                        freq[idx_rem] -= 1
                        
                        # Update counters based on the removed character
                        if old_freq_rem == 1:
                            unique_chars -= 1
                        if old_freq_rem == k + 1:
                            chars_at_k += 1
                        elif old_freq_rem == k:
                            chars_at_k -= 1
                    
                    # --- Check if the current window forms a complete substring ---
                    if i >= window_len - 1:
                        # A window is complete if it has `u` unique characters,
                        # and all of them appear exactly `k` times.
                        if unique_chars == u and chars_at_k == u:
                            count += 1
                            
            return count

        n = len(word)
        total_complete = 0
        
        # Split the word into segments where adjacent characters have a diff <= 2
        start = 0
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i - 1])) > 2:
                # A boundary is found, process the segment before it
                total_complete += _count_in_segment(word[start:i])
                start = i
        
        # Process the last segment (or the whole word if no boundaries were found)
        total_complete += _count_in_segment(word[start:n])
        
        return total_complete