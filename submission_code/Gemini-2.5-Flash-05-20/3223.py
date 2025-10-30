import collections

class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        total_complete_substrings = 0

        # Step 1: Split the word into segments based on the alphabetical proximity condition.
        # Any complete substring cannot cross a boundary where abs(ord(c1) - ord(c2)) > 2.
        segments = []
        current_segment_start = 0
        for i in range(1, n):
            # If the absolute difference between adjacent characters is greater than 2,
            # it marks a boundary for complete substrings.
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                segments.append(word[current_segment_start:i])
                current_segment_start = i
        # Add the last segment (or the only segment if no splits occurred)
        segments.append(word[current_segment_start:])

        # Step 2: Process each segment independently using a sliding window.
        for segment_str in segments:
            segment_len = len(segment_str)

            # Iterate through possible number of distinct characters 'm' (from 1 to 26).
            # A complete substring must have length 'm * k'.
            for m in range(1, 27):
                target_len = m * k
                
                # If the required length for 'm' distinct characters exceeds the current segment's length,
                # then no complete substring with 'm' or more distinct characters can be formed in this segment.
                # Since 'target_len' increases with 'm', we can break and move to the next segment.
                if target_len > segment_len:
                    break 

                freq = collections.defaultdict(int)
                distinct_count = 0      # Number of distinct characters in the current window
                count_exact_k = 0       # Number of distinct characters whose frequency is exactly 'k'

                # Initialize the first window of size target_len
                for j in range(target_len):
                    char = segment_str[j]
                    
                    # If this character is seen for the first time in the window, increment distinct_count.
                    if freq[char] == 0:
                        distinct_count += 1
                    
                    # If this character's frequency was 'k' (and is about to become 'k+1'), decrement count_exact_k.
                    if freq[char] == k:
                        count_exact_k -= 1
                    
                    freq[char] += 1
                    
                    # If this character's frequency just became 'k', increment count_exact_k.
                    if freq[char] == k:
                        count_exact_k += 1
                
                # Check the first window for completeness criteria:
                # 1. The number of distinct characters in the window must be 'm'.
                # 2. All 'm' of these distinct characters must have a frequency of exactly 'k'.
                if distinct_count == m and count_exact_k == m:
                    total_complete_substrings += 1

                # Slide the window across the segment
                for i in range(1, segment_len - target_len + 1):
                    # Remove the character leaving the window from the left (at index i-1)
                    left_char = segment_str[i-1]
                    
                    # If left_char's frequency was 'k' (and is about to become 'k-1'), decrement count_exact_k.
                    if freq[left_char] == k:
                        count_exact_k -= 1
                    
                    freq[left_char] -= 1
                    
                    # If left_char's frequency just became 0, decrement distinct_count (it's no longer in the window).
                    if freq[left_char] == 0:
                        distinct_count -= 1
                    
                    # If left_char's frequency was 'k+1' and just became 'k', increment count_exact_k.
                    if freq[left_char] == k:
                        count_exact_k += 1

                    # Add the character entering the window from the right (at index i + target_len - 1)
                    right_char = segment_str[i + target_len - 1]
                    
                    # If right_char is new to the window, increment distinct_count.
                    if freq[right_char] == 0:
                        distinct_count += 1
                    
                    # If right_char's frequency was 'k' (and is about to become 'k+1'), decrement count_exact_k.
                    if freq[right_char] == k:
                        count_exact_k -= 1
                    
                    freq[right_char] += 1
                    
                    # If right_char's frequency just became 'k', increment count_exact_k.
                    if freq[right_char] == k:
                        count_exact_k += 1
                    
                    # Check the current window for completeness criteria
                    if distinct_count == m and count_exact_k == m:
                        total_complete_substrings += 1
        
        return total_complete_substrings