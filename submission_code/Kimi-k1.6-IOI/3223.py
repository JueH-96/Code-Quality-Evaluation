class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        if not word:
            return 0
        
        # Split the word into segments where adjacent characters have difference <= 2
        segments = []
        current = [word[0]]
        for i in range(1, len(word)):
            prev_char = word[i-1]
            curr_char = word[i]
            prev = ord(prev_char) - ord('a')
            curr = ord(curr_char) - ord('a')
            if abs(curr - prev) > 2:
                segments.append(''.join(current))
                current = [curr_char]
            else:
                current.append(curr_char)
        segments.append(''.join(current))
        
        total = 0
        
        for seg in segments:
            n = len(seg)
            for m in range(1, 27):  # m can be from 1 to 26
                window = m * k
                if window > n:
                    continue
                freq = [0] * 26
                count_k = 0
                left = 0
                valid = 0
                for right in range(n):
                    # Add the current character
                    c = seg[right]
                    idx = ord(c) - ord('a')
                    prev_freq = freq[idx]
                    freq[idx] += 1
                    if prev_freq == k - 1:
                        count_k += 1
                    elif prev_freq == k:
                        count_k -= 1
                    
                    # Maintain window size <= window
                    while (right - left + 1) > window:
                        # Remove the leftmost character
                        lc = seg[left]
                        l_idx = ord(lc) - ord('a')
                        l_prev_freq = freq[l_idx]
                        freq[l_idx] -= 1
                        if l_prev_freq == k:
                            count_k -= 1
                        elif l_prev_freq == k + 1:
                            count_k += 1
                        left += 1
                    
                    # Check if current window is valid
                    if (right - left + 1) == window:
                        if count_k == m:
                            valid += 1
                total += valid
        
        return total