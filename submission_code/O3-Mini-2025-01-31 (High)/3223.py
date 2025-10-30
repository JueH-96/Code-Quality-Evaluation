class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        # The idea is to count those substrings that satisfy two conditions:
        # 1. They are "smooth": every adjacent pair of characters has an absolute alphabetical difference ≤ 2.
        # 2. They are "complete": if the substring contains m distinct letters (its length is m*k),
        #    then each letter in that substring appears exactly k times.
        #
        # Our plan is to first split the word into segments where the adjacent difference condition holds.
        # (Any substring that spans a gap where the condition fails cannot be valid.)
        # Then for each such segment (of length seg_len), we consider windows whose length 
        # is a multiple of k (say, win_len = m * k, and clearly m <= 26 since the alphabet has 26 letters)
        # and we count, with a sliding window, how many windows satisfy that every letter appears exactly k times.
        #
        # Because in a valid window of length m*k, we must have exactly m distinct letters (each appearing k times),
        # we can maintain a frequency array and track both the number of distinct letters and the count
        # of letters that currently have exactly k occurrences.
        #
        # This sliding window (with fixed length) approach is efficient because for each possible m (which is at most 26)
        # we slide over the segment (whose total length is at most len(word)); since len(word) is 10^5,
        # the total inner‐loop iterations are not too many.
        
        n = len(word)
        ans = 0
        base = ord('a')
        i = 0
        
        # Split the word into segments where every adjacent difference is ≤ 2.
        while i < n:
            seg_start = i
            while i < n - 1 and abs(ord(word[i+1]) - ord(word[i])) <= 2:
                i += 1
            seg_end = i
            seg = word[seg_start: seg_end + 1]
            seg_len = len(seg)
            
            # For a substring to be complete, its length must be m*k for some m (the number of distinct letters),
            # and m cannot exceed 26.
            max_m = seg_len // k
            if max_m > 26:
                max_m = 26
                
            # For each possible m (distinct letter count), check all windows of length m*k.
            for m in range(1, max_m + 1):
                win_len = m * k
                if win_len > seg_len:
                    break
                # Use a sliding window of fixed size = win_len
                freq = [0] * 26  # frequency of each letter
                distinct = 0     # count of letters that appear (nonzero frequency)
                exactly = 0      # count of letters with frequency exactly k
                
                # Initialize the first window in the segment.
                for j in range(win_len):
                    idx = ord(seg[j]) - base
                    old = freq[idx]
                    new = old + 1
                    freq[idx] = new
                    if old == 0:
                        distinct += 1
                    if new == k:
                        exactly += 1
                    if new == k + 1:
                        exactly -= 1
                if distinct == m and exactly == m:
                    ans += 1
                
                # Slide the window over the segment.
                for start in range(1, seg_len - win_len + 1):
                    # Remove the left-most character from the old window.
                    left_idx = ord(seg[start - 1]) - base
                    old = freq[left_idx]
                    new = old - 1
                    freq[left_idx] = new
                    if old == 1:
                        distinct -= 1
                    if old == k:
                        exactly -= 1
                    if old == k + 1:
                        exactly += 1
                    
                    # Add the new right-most character.
                    right_idx = ord(seg[start + win_len - 1]) - base
                    old = freq[right_idx]
                    new = old + 1
                    freq[right_idx] = new
                    if old == 0:
                        distinct += 1
                    if new == k:
                        exactly += 1
                    if new == k + 1:
                        exactly -= 1
                    
                    if distinct == m and exactly == m:
                        ans += 1
            i = seg_end + 1
        return ans

# For example testing:
if __name__=="__main__":
    sol = Solution()
    # Example 1
    print(sol.countCompleteSubstrings("igigee", 2))  # Expected output: 3
    # Example 2
    print(sol.countCompleteSubstrings("aaabbbccc", 3))  # Expected output: 6
    # Some additional tests:
    print(sol.countCompleteSubstrings("abz", 1))  # "abz" has valid substrings: "a", "b", "z", "ab" so output should be 4