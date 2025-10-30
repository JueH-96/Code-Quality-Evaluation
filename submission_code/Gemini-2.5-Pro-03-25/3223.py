class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        """
        Counts the number of complete substrings in the given word.
        A substring `s` is complete if:
        1. Each character in `s` appears exactly k times.
        2. The difference between two adjacent characters in `word` that are part of `s` is at most 2.
           This effectively means we only need to consider substrings within segments of `word` where all adjacent characters satisfy `abs(ord(c1) - ord(c2)) <= 2`.

        Args:
          word: The input string consisting of lowercase English letters.
          k: The required frequency for each character in a complete substring.

        Returns:
          The total count of complete substrings.
        """
        n = len(word)
        total_complete_substrings = 0
        start = 0
        while start < n:
            end = start
            # Find the end of the current segment where adjacent character difference is at most 2
            # A segment ends when this condition is violated or we reach the end of the string.
            while end + 1 < n and abs(ord(word[end]) - ord(word[end + 1])) <= 2:
                end += 1
            
            # Process the identified segment word[start : end+1]
            segment = word[start : end + 1]
            # Use a helper method to count complete substrings within this segment
            total_complete_substrings += self._count_complete_in_segment(segment, k)
            
            # Move to the start of the next potential segment
            start = end + 1
            
        return total_complete_substrings

    def _count_complete_in_segment(self, segment: str, k: int) -> int:
        """
        Helper method to count complete substrings within a segment using a sliding window approach.
        This method assumes the segment already satisfies the adjacent character difference condition.
        It checks the condition that each character within a substring appears exactly k times.
        
        Args:
            segment: The segment string (substring of the original word).
            k: The required frequency for each character.
            
        Returns:
            The count of complete substrings within the given segment.
        """
        count = 0
        N = len(segment)
        # Minimum possible length of a complete substring is k (when there's only 1 distinct character).
        # If segment length is less than k, no complete substring is possible.
        if N < k: 
            return 0
            
        # Iterate through possible numbers of distinct characters (d) in a complete substring.
        # Since there are 26 lowercase English letters, d can range from 1 to 26.
        for d in range(1, 27): 
            # Calculate the required length (L) for a complete substring with d distinct characters.
            # Each of the d characters must appear k times, so L = d * k.
            L = d * k 
            if L > N:
                # If the required length exceeds the segment length, we can stop checking larger d values,
                # as the length L will only increase.
                break 
            
            # Use a sliding window of size L over the segment.
            counts = [0] * 26 # Frequency map for characters 'a' through 'z' in the current window.
            
            # bad_counts tracks the number of character types currently in the window 
            # whose frequency is NOT equal to k (but is > 0).
            # A window represents a complete substring if bad_counts is 0.
            bad_counts = 0 

            # Initialize counts and bad_counts for the first window: segment[0:L]
            # This initialization takes O(L) time.
            for i in range(L):
                char_idx = ord(segment[i]) - ord('a')
                
                # Before incrementing counts[char_idx]:
                # If the current count is > 0 and not k, it means this character type was counted as "bad".
                # Since its count is about to change, we decrement bad_counts temporarily.
                if counts[char_idx] > 0 and counts[char_idx] != k:
                    bad_counts -= 1

                counts[char_idx] += 1 # Increment the frequency count for the character.
                
                # After incrementing counts[char_idx]:
                # If the new count is > 0 and not k, this character type is now "bad". Increment bad_counts.
                if counts[char_idx] > 0 and counts[char_idx] != k:
                    bad_counts += 1

            # Check if the first window (segment[0:L]) is complete.
            if bad_counts == 0:
                count += 1
            
            # Slide the window one character at a time, from index 1 up to N - L.
            # This loop runs N - L times. Each iteration updates counts and bad_counts in O(1) time.
            for i in range(1, N - L + 1):
                # --- Remove character segment[i-1] (char_out) from the left end of the window ---
                char_out_idx = ord(segment[i-1]) - ord('a')
                
                # Before decrementing counts[char_out_idx]:
                # If its count was > 0 and not k, it was "bad". Decrement bad_counts.
                if counts[char_out_idx] > 0 and counts[char_out_idx] != k:
                    bad_counts -= 1
                
                counts[char_out_idx] -= 1 # Decrement the count for the character leaving the window.
                
                # After decrementing counts[char_out_idx]:
                # If its new count is > 0 and not k, it is now "bad". Increment bad_counts.
                if counts[char_out_idx] > 0 and counts[char_out_idx] != k:
                    bad_counts += 1

                # --- Add character segment[i+L-1] (char_in) to the right end of the window ---
                char_in_idx = ord(segment[i+L-1]) - ord('a')
                
                # Before incrementing counts[char_in_idx]:
                # If its count was > 0 and not k, it was "bad". Decrement bad_counts.
                if counts[char_in_idx] > 0 and counts[char_in_idx] != k:
                    bad_counts -= 1
            
                counts[char_in_idx] += 1 # Increment the count for the character entering the window.
                
                # After incrementing counts[char_in_idx]:
                # If its new count is > 0 and not k, it is now "bad". Increment bad_counts.
                if counts[char_in_idx] > 0 and counts[char_in_idx] != k:
                    bad_counts += 1
                
                # Check if the current window (segment[i : i+L]) is complete.
                # If bad_counts is 0, all present characters have frequency k.
                if bad_counts == 0:
                    count += 1
                    
        # Return the total count of complete substrings found in this segment.
        return count