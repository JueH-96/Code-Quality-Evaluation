class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s, change_idx=-1, change_char=''):
            partitions = 0
            i = 0
            n = len(s)
            
            while i < n:
                distinct_chars = set()
                j = i
                
                # Find longest prefix with at most k distinct characters
                while j < n:
                    char = change_char if j == change_idx else s[j]
                    distinct_chars.add(char)
                    if len(distinct_chars) > k:
                        break
                    j += 1
                
                partitions += 1
                i = j
            
            return partitions
        
        # Try no change first
        max_partitions = count_partitions(s)
        
        # Try changing each position to each possible character
        for i in range(len(s)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != s[i]:  # Only try if it's actually a change
                    max_partitions = max(max_partitions, count_partitions(s, i, c))
        
        return max_partitions