class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        """
        We want to count how many substrings have:
          1) Each character in the substring occurs exactly k times.
          2) The difference between two adjacent characters (by alphabet index) is at most 2.
        
        Strategy:
        1) First, split the string into blocks (substrings) where for each block, 
           consecutive characters differ by at most 2. No valid substring can cross 
           the boundary where the adjacency condition fails.
        
        2) For each such "adjacency block", use a two-pointer sliding-window approach
           to count how many sub-substrings (fully contained in that block) have 
           each distinct character appearing exactly k times.

           - We'll maintain a frequency dictionary freq of size 26 (for 'a'..'z') in
             the current window.
           - distinctCount = number of characters with freq > 0 in the window.
           - countK = number of characters whose freq == k in the window.
           - We expand the right pointer, increment the frequency for that character,
             update distinctCount and countK as needed.
           - We shrink the window from the left while either 
               (a) any character has freq > k (can't be valid), or 
               (b) the total window size is larger than distinctCount * k 
                   (then it can't have each distinct char = k because the window is too big).
           - After adjusting, if the window size == distinctCount * k and countK == distinctCount,
             then we've found exactly one valid substring ending at right (since for a fixed right,
             there's at most one left that makes window size == distinctCount*k).
        
        Time Complexity:
          Splitting into blocks is O(n). The two-pointer approach in each block is linear 
          in the size of that block. Summed over all blocks, this is O(n) overall.
        """

        # Helper function to count valid substrings in a block that satisfies adjacency
        def count_in_block(s: str, k: int) -> int:
            freq = [0]*26
            left = 0
            distinct_count = 0
            count_k = 0
            res = 0

            for right, ch in enumerate(s):
                idx = ord(ch) - ord('a')
                freq[idx] += 1
                
                # If this character frequency became 1, we have a new distinct character
                if freq[idx] == 1:
                    distinct_count += 1
                # If this character frequency just hit k
                if freq[idx] == k:
                    count_k += 1
                # If it exceeded k, it can't be valid
                while freq[idx] > k or (right - left + 1) > distinct_count * k:
                    left_idx = ord(s[left]) - ord('a')
                    
                    # If that left char was exactly k before removing it, count_k decreases
                    if freq[left_idx] == k:
                        count_k -= 1
                    freq[left_idx] -= 1
                    # If frequency of left char is now 0, we lost a distinct character
                    if freq[left_idx] == 0:
                        distinct_count -= 1
                    left += 1
                
                # Now check if the current window is a valid substring
                window_size = right - left + 1
                if window_size == distinct_count * k and count_k == distinct_count:
                    res += 1
            
            return res

        n = len(word)
        if n == 0:
            return 0
        
        total = 0
        block_start = 0
        
        # Split into adjacency blocks
        for i in range(1, n):
            # If adjacency fails, process the block [block_start..i-1], then start a new block
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                total += count_in_block(word[block_start:i], k)
                block_start = i
        
        # Don't forget the last block
        total += count_in_block(word[block_start:], k)
        
        return total