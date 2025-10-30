class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def countPartitions(s):
            if not s:
                return 0
            
            partitions = 0
            i = 0
            n = len(s)
            
            while i < n:
                distinct = set()
                j = i
                
                # Find the longest prefix starting at i with at most k distinct chars
                while j < n:
                    distinct.add(s[j])
                    if len(distinct) > k:
                        break
                    j += 1
                
                partitions += 1
                i = j
            
            return partitions
        
        # Try without changing any character
        max_partitions = countPartitions(s)
        
        # Try changing each position to each possible character
        n = len(s)
        s_list = list(s)
        
        for i in range(n):
            original = s_list[i]
            
            # Try changing to each letter
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != original:
                    s_list[i] = c
                    max_partitions = max(max_partitions, countPartitions(''.join(s_list)))
                    s_list[i] = original
        
        return max_partitions