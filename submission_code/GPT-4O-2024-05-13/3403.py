class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        from collections import Counter
        
        def is_balanced(sub):
            count = Counter(sub)
            return len(set(count.values())) == 1
        
        n = len(s)
        min_partitions = n
        
        for i in range(1, n + 1):
            partitions = []
            start = 0
            while start < n:
                for end in range(start + i, n + 1):
                    if is_balanced(s[start:end]):
                        partitions.append(s[start:end])
                        start = end
                        break
                else:
                    break
            if start == n:
                min_partitions = min(min_partitions, len(partitions))
        
        return min_partitions