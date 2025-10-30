class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        def is_balanced(sub):
            count = {}
            for char in sub:
                if char in count:
                    count[char] += 1
                else:
                    count[char] = 1
            return all(value % len(sub) == 0 for value in count.values())
        
        def dfs(start):
            if start == len(s):
                return 0
            min_partitions = float('inf')
            for end in range(start + 1, len(s) + 1):
                if is_balanced(s[start:end]):
                    min_partitions = min(min_partitions, 1 + dfs(end))
            return min_partitions
        
        return dfs(0)