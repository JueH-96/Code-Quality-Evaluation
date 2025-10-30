import string

class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        # Compute the number of partitions with no change
        max_part = self._min_partition(s, k, -1, 'a')  # change_p = -1 means no change
        
        # Find the set of used characters
        used = set(s)
        
        # If there is a character not in s, try changing each position to that character
        if len(used) < 26:
            # Find a character not in used
            for ch in string.ascii_lowercase:
                if ch not in used:
                    B_not_in = ch
                    break
            # Try changing each position p to B_not_in
            for p in range(n):
                part = self._min_partition(s, k, p, B_not_in)
                if part > max_part:
                    max_part = part
        
        return max_part
    
    def _min_partition(self, s: str, k: int, change_p: int, change_b: str) -> int:
        n = len(s)
        i = 0
        partitions = 0
        while i < n:
            # Find the farthest j such that s[i..j] has at most k distinct characters with the change
            freq = [0] * 26
            d = 0
            for pos in range(i, n):
                # Get the character index, considering the change
                if pos == change_p:
                    idx_char = ord(change_b)
                else:
                    idx_char = ord(s[pos])
                char_idx = idx_char - ord('a')
                if freq[char_idx] == 0 and d == k:
                    # Cannot add this character, break
                    j_end = pos - 1
                    break
                if freq[char_idx] == 0:
                    d += 1
                freq[char_idx] += 1
            else:
                # No break, the end is n-1
                j_end = n - 1
            # Increment partition count and move to the next starting point
            partitions += 1
            i = j_end + 1
        return partitions