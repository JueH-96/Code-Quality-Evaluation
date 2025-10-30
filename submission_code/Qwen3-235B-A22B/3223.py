class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        segments = []
        n = len(word)
        if n == 0:
            return 0
        start = 0
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                segments.append(word[start:i])
                start = i
        segments.append(word[start:])
        
        res = 0
        
        for seg in segments:
            L = len(seg)
            for m in range(1, 27):
                ws = m * k
                if ws > L:
                    continue
                # Sliding window setup
                freq = [0] * 26
                valid = 0
                count = 0
                # Initialize first window
                for i in range(ws):
                    c = seg[i]
                    idx = ord(c) - ord('a')
                    prev = freq[idx]
                    if prev == k:
                        valid -= 1
                    if prev == k - 1:
                        valid += 1
                    freq[idx] += 1
                if valid == m:
                    count += 1
                # Slide window
                for i in range(ws, L):
                    # Remove left character
                    left_char = seg[i - ws]
                    idx_l = ord(left_char) - ord('a')
                    prev = freq[idx_l]
                    if prev == k:
                        valid -= 1
                    if prev == k + 1:
                        valid += 1
                    freq[idx_l] -= 1
                    # Add right character
                    right_char = seg[i]
                    idx_r = ord(right_char) - ord('a')
                    prev_r = freq[idx_r]
                    if prev_r == k:
                        valid -= 1
                    if prev_r == k - 1:
                        valid += 1
                    freq[idx_r] += 1
                    # Check valid
                    if valid == m:
                        count += 1
                res += count
        return res