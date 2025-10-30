class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        # Convert the string to a list of integers representing each character
        ord_s = [ord(c) - ord('a') for c in s]
        n = len(ord_s)
        max_partitions = self._count_partitions(ord_s, k)
        
        # Try changing each character
        for i in range(n):
            original = ord_s[i]
            for new_char in range(26):
                if new_char == original:
                    continue
                # Modify the i-th character
                ord_s[i] = new_char
                # Compute the partitions
                current = self._count_partitions(ord_s, k)
                if current > max_partitions:
                    max_partitions = current
                # Revert the change
                ord_s[i] = original
        
        return max_partitions
    
    def _count_partitions(self, ord_s, k):
        count = 0
        left = 0
        n = len(ord_s)
        while left < n:
            freq = [0] * 26
            distinct = 0
            right = left
            while right < n:
                idx = ord_s[right]
                if freq[idx] == 0:
                    distinct += 1
                    if distinct > k:
                        break
                freq[idx] += 1
                right += 1
            count += 1
            left = right
        return count