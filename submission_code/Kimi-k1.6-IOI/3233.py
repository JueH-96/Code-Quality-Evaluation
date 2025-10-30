class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        s_bytes = [ord(c) - ord('a') for c in s]
        n = len(s_bytes)
        max_partitions = 1
        
        def compute_partitions(change_pos, change_char):
            mask = 0
            count = 0
            res = 0
            for i in range(n):
                c = s_bytes[i]
                if i == change_pos:
                    c = change_char
                bit = 1 << c
                if not (mask & bit):
                    if count == k:
                        res += 1
                        mask = bit
                        count = 1
                    else:
                        mask |= bit
                        count += 1
            return res + 1
        
        original = compute_partitions(-1, -1)
        max_partitions = original
        
        for i in range(n):
            current_char = s_bytes[i]
            for c in range(26):
                if c == current_char:
                    continue
                partitions = compute_partitions(i, c)
                if partitions > max_partitions:
                    max_partitions = partitions
                    # Early exit if max possible partitions are found
                    if max_partitions == n:
                        return max_partitions
        
        return max_partitions