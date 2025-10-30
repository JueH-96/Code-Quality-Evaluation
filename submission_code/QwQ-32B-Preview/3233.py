class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Function to compute number of partitions for s[l..r]
        def compute_partitions(l, r):
            parts = 0
            distinct = 0
            freq = [0] * 26
            start = l
            for i in range(l, r + 1):
                idx = ord(s[i]) - ord('a')
                if freq[idx] == 0:
                    distinct += 1
                freq[idx] += 1
                if distinct > k:
                    parts += 1
                    # Reset for the next partition
                    distinct = 1
                    freq = [0] * 26
                    freq[idx] = 1
                    start = i
            if start <= r:
                parts += 1
            return parts
        
        # Original partitions without any changes
        total_parts = compute_partitions(0, n - 1)
        
        # Try changing one character to maximize partitions
        max_additional_parts = 0
        for j in range(n):
            original_char = s[j]
            for d in range(26):
                new_char = chr(ord('a') + d)
                if new_char == original_char:
                    continue
                # Modify s[j] to new_char and compute partitions
                s_list = list(s)
                s_list[j] = new_char
                modified_s = ''.join(s_list)
                parts = compute_partitions(0, n - 1)
                max_additional_parts = max(max_additional_parts, parts)
        
        # The final answer is the maximum of original parts and parts after one change
        return max(total_parts, max_additional_parts)