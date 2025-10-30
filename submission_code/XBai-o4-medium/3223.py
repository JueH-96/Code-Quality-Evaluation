class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        if not word:
            return 0
        
        # Split the word into groups where adjacent characters differ by at most 2
        groups = []
        current = [word[0]]
        prev_char = word[0]
        for c in word[1:]:
            if abs(ord(c) - ord(prev_char)) <= 2:
                current.append(c)
            else:
                groups.append(current)
                current = [c]
            prev_char = c
        groups.append(current)
        
        def count_valid(s, k):
            n = len(s)
            if n < k:
                return 0
            total = 0
            max_m = 26  # maximum possible distinct characters
            for m in range(1, max_m + 1):
                window_len = m * k
                if window_len > n:
                    break
                # Sliding window of length window_len
                freq = [0] * 26
                # Initialize the first window
                for i in range(window_len):
                    freq[ord(s[i]) - ord('a')] += 1
                # Check validity of the first window
                valid = True
                for cnt in freq:
                    if cnt not in (0, k):
                        valid = False
                        break
                if valid:
                    total += 1
                # Slide the window across the string
                for i in range(1, n - window_len + 1):
                    # Remove the character exiting the window
                    out_char = s[i - 1]
                    freq[ord(out_char) - ord('a')] -= 1
                    # Add the new character entering the window
                    in_char = s[i + window_len - 1]
                    freq[ord(in_char) - ord('a')] += 1
                    # Check validity
                    valid = True
                    for cnt in freq:
                        if cnt not in (0, k):
                            valid = False
                            break
                    if valid:
                        total += 1
            return total
        
        # Sum the valid substrings from all groups
        res = 0
        for group in groups:
            res += count_valid(group, k)
        return res