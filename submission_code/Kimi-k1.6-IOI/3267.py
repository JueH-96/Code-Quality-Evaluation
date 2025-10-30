class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict
        
        # Create groups of consecutive characters
        groups = []
        n = len(s)
        if n == 0:
            return -1
        
        current_char = s[0]
        count = 1
        for i in range(1, n):
            if s[i] == current_char:
                count += 1
            else:
                groups.append((current_char, count))
                current_char = s[i]
                count = 1
        groups.append((current_char, count))  # Add the last group
        
        # Initialize counts for each character and length
        counts = defaultdict(lambda: defaultdict(int))
        for char, length in groups:
            for l in range(1, length + 1):
                counts[char][l] += (length - l + 1)
        
        max_len = -1
        # Check all counts to find the maximum valid length
        for char in counts:
            for l in counts[char]:
                if counts[char][l] >= 3:
                    if l > max_len:
                        max_len = l
        
        return max_len if max_len != -1 else -1