class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        counts = {}
        left = 0
        max_len = 0
        for right in range(len(s)):
            c = s[right]
            counts[c] = counts.get(c, 0) + 1
            # Check if current character's count exceeds 2
            while counts[c] > 2:
                left_c = s[left]
                counts[left_c] -= 1
                if counts[left_c] == 0:
                    del counts[left_c]
                left += 1
            # Update the maximum length
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len