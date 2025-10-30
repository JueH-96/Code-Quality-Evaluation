class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        if n == 0:
            return 0

        # First, break the word into maximal "smooth" segments,
        # i.e. segments in which adjacent characters have absolute difference at most 2.
        segments = []
        seg_start = 0
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                segments.append((seg_start, i-1))
                seg_start = i
        segments.append((seg_start, n-1))
        
        total_count = 0
        
        # For each segment, we want to count how many substrings are "complete".
        # A substring s is complete if:
        #   1. For each character in s, its frequency is exactly k.
        #   2. (By our segmentation, the adjacent differences condition is already guaranteed
        #      since any substring staying inside a smooth segment inherits the property.)
        #
        # Observation: if s is complete and has length L, then every character that appears in s
        # appears exactly k times. Hence, if there are t distinct characters in s, then L = t*k.
        # Thus, for a given t (number of distinct characters) from 1 to min(26, L//k),
        # we only need to check fixed-length windows of size t*k within the segment.
        
        # We'll iterate over each segment:
        for seg_start, seg_end in segments:
            seg = word[seg_start : seg_end+1]
            L = len(seg)
            # For each possible number of distinct characters t
            # such that window length = t*k is no more than L.
            for t in range(1, min(26, L // k) + 1):
                window_size = t * k
                # sliding window over seg with fixed window_size
                # frequency array for 26 letters
                freq = [0] * 26
                distinct = 0  # count distinct letters in window
                count_exact = 0  # count of letters that appear exactly k times

                # Function to get index from char
                def char_index(ch):
                    return ord(ch) - ord('a')

                # Initialize the window [0, window_size-1]
                for i in range(window_size):
                    idx = char_index(seg[i])
                    before = freq[idx]
                    freq[idx] += 1
                    after = freq[idx]
                    if before == 0:
                        distinct += 1
                    # When adding, if frequency becomes exactly k, then add;
                    # but if it was already k, then it leaves the exactly-k condition.
                    if after == k:
                        count_exact += 1
                    if before == k:
                        count_exact -= 1

                # Now, check this first window.
                if distinct == t and count_exact == t:
                    total_count += 1

                # Slide the window along segment.
                for i in range(window_size, L):
                    # Remove the leftmost character from position i - window_size and add seg[i]
                    left_char = seg[i - window_size]
                    right_char = seg[i]
                    left_idx = char_index(left_char)
                    right_idx = char_index(right_char)
                    
                    # Remove left_char:
                    before = freq[left_idx]
                    freq[left_idx] -= 1
                    after = freq[left_idx]
                    # If the frequency was exactly k then after removal it becomes k-1: no longer complete.
                    if before == k:
                        count_exact -= 1
                    # Special case: if frequency was k+1, then after removal it becomes exactly k.
                    if before == k + 1:
                        count_exact += 1
                    # If character completely removed, update distinct count.
                    if freq[left_idx] == 0:
                        distinct -= 1

                    # Add right_char:
                    before = freq[right_idx]
                    freq[right_idx] += 1
                    after = freq[right_idx]
                    if before == 0:
                        distinct += 1
                    if after == k:
                        count_exact += 1
                    if before == k:
                        count_exact -= 1

                    # Check current window condition.
                    if distinct == t and count_exact == t:
                        total_count += 1

        return total_count

# For local testing, you can use the following code:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    word1 = "igigee"
    k1 = 2
    print(sol.countCompleteSubstrings(word1, k1))  # Expected output: 3

    # Example 2:
    word2 = "aaabbbccc"
    k2 = 3
    print(sol.countCompleteSubstrings(word2, k2))  # Expected output: 6