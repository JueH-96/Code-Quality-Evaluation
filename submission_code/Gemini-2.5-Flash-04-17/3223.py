from collections import defaultdict

class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        total_count = 0
        start = 0

        # Step 1: Split the word into segments based on the adjacent character difference condition
        # A substring is complete only if all its adjacent characters have an absolute difference
        # in alphabetical position of at most 2. If word[i] and word[i-1] violate this,
        # no complete substring can span across this boundary.
        for i in range(1, n):
            # Check if the absolute difference in alphabetical position between word[i] and word[i-1] is greater than 2
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                # If the condition is not met, the substring word[start:i] forms a segment
                segment = word[start:i]
                # Process this segment to count complete substrings within it
                total_count += self.count_complete_in_segment(segment, k)
                # Start a new segment from the current index i
                start = i

        # Process the last segment from 'start' to the end of the word
        segment = word[start:n]
        total_count += self.count_complete_in_segment(segment, k)

        return total_count

    # Helper function to count complete substrings within a single segment
    # Within a segment, the adjacent character difference condition is guaranteed.
    # We only need to find substrings where each character appears exactly k times.
    # Such a substring must have a length that is a multiple of k.
    # If a substring has m distinct characters, its length must be m * k.
    def count_complete_in_segment(self, segment: str, k: int) -> int:
        seg_len = len(segment)
        segment_count = 0

        # Iterate through possible number of distinct characters (m) in a complete substring
        # A complete substring with m distinct characters must have a total length of m * k
        # The maximum number of distinct characters is 26 (lowercase alphabet)
        for m in range(1, 27):
            window_size = m * k

            # If the minimum possible length for a complete substring with m distinct chars
            # is greater than the current segment length, we cannot find any such substring
            # with m or more distinct chars. Break the inner loop.
            if window_size > seg_len:
                break

            # Use a sliding window of fixed size 'window_size'
            freq = [0] * 26 # Frequency array for characters 'a' through 'z' in the current window
            distinct_count = 0     # Number of unique characters with frequency > 0 in the window
            chars_with_freq_k = 0  # Number of unique characters with frequency == k in the window
            
            base_ord = ord('a')

            # --- Initialize the first window [0, window_size - 1] ---
            for i in range(window_size):
                char_idx = ord(segment[i]) - base_ord
                
                # Update distinct_count: check if character was not present before adding
                if freq[char_idx] == 0:
                    distinct_count += 1
                # Update chars_with_freq_k: check if character was k before adding (transition k -> k+1)
                if freq[char_idx] == k:
                    chars_with_freq_k -= 1

                # Increment frequency
                freq[char_idx] += 1

                # Update chars_with_freq_k: check if character is k after adding (transition k-1 -> k)
                if freq[char_idx] == k:
                    chars_with_freq_k += 1

            # Check if the first window is a complete substring
            # A window of size m*k is complete if it has exactly m distinct characters
            # AND all m distinct characters have a frequency of exactly k.
            # This condition is equivalent to: distinct_count == m AND chars_with_freq_k == distinct_count.
            if distinct_count == m and chars_with_freq_k == distinct_count:
                 segment_count += 1

            # --- Slide the window from index 1 up to seg_len - window_size ---
            # The window slides from [i-1, i-1+window_size-1] to [i, i+window_size-1]
            for i in range(1, seg_len - window_size + 1):
                # Character c_out at index i-1 leaves the window
                char_out_idx = ord(segment[i-1]) - base_ord
                
                # Update chars_with_freq_k: check if character was k before removing (transition k -> k-1)
                if freq[char_out_idx] == k:
                    chars_with_freq_k -= 1
                # Update chars_with_freq_k: check if character was k+1 before removing (transition k+1 -> k)
                if freq[char_out_idx] == k + 1: 
                    chars_with_freq_k += 1

                # Decrement frequency
                freq[char_out_idx] -= 1
                
                # Update distinct_count: check if character is 0 after removing (transition 1 -> 0)
                if freq[char_out_idx] == 0:
                    distinct_count -= 1


                # Character c_in at index i + window_size - 1 enters the window
                char_in_idx = ord(segment[i + window_size - 1]) - base_ord

                # Update distinct_count: check if character was 0 before adding (transition 0 -> 1)
                if freq[char_in_idx] == 0:
                    distinct_count += 1
                # Update chars_with_freq_k: check if character was k-1 before adding (transition k-1 -> k)
                if freq[char_in_idx] == k - 1:
                    chars_with_freq_k += 1
                # Update chars_with_freq_k: check if character was k before adding (transition k -> k+1)
                if freq[char_in_idx] == k: 
                     chars_with_freq_k -= 1

                # Increment frequency
                freq[char_in_idx] += 1

                # Check if the current window [i, i + window_size - 1] is a complete substring
                if distinct_count == m and chars_with_freq_k == distinct_count:
                     segment_count += 1

        return segment_count