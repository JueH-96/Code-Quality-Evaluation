class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        original = self.compute_partitions(s, k)
        max_partitions = original
        n = len(s)
        
        for i in range(n):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == s[i]:
                    continue
                modified = s[:i] + c + s[i+1:]
                current = self.compute_partitions(modified, k)
                if current > max_partitions:
                    max_partitions = current
        return max_partitions
    
    def compute_partitions(self, s, k):
        count = 0
        start = 0
        freq = {}
        distinct = 0
        previous_end = -1
        
        for end in range(len(s)):
            c = s[end]
            if freq.get(c, 0) == 0:
                distinct += 1
            freq[c] = freq.get(c, 0) + 1
            
            while distinct > k:
                sc = s[start]
                freq[sc] -= 1
                if freq[sc] == 0:
                    distinct -= 1
                start += 1
            
            # Check if the current window starts after the previous partition's end
            if start > previous_end:
                count += 1
                previous_end = end
        
        return count