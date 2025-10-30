class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        if n == 1:
            return 1
        
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        
        if all(x == count[0] for x in count):
            return 1
        
        min_partitions = float('inf')
        for i in range(1, n):
            left_count = [0] * 26
            right_count = [0] * 26
            
            for j in range(i):
                left_count[ord(s[j]) - ord('a')] += 1
            
            for j in range(i, n):
                right_count[ord(s[j]) - ord('a')] += 1
            
            if all(x == left_count[0] for x in left_count) and all(x == right_count[0] for x in right_count):
                min_partitions = min(min_partitions, 2)
        
        if min_partitions == float('inf'):
            return n
        else:
            return min_partitions