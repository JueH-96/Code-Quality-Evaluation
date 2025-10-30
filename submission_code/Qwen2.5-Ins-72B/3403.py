class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        def is_balanced(subs):
            count = {}
            for char in subs:
                if char in count:
                    count[char] += 1
                else:
                    count[char] = 1
            return len(set(count.values())) == 1
        
        def min_partitions(s, start, memo):
            if start == len(s):
                return 0
            if start in memo:
                return memo[start]
            
            min_partitions_count = float('inf')
            for end in range(start + 1, len(s) + 1):
                if is_balanced(s[start:end]):
                    min_partitions_count = min(min_partitions_count, 1 + min_partitions(s, end, memo))
            
            memo[start] = min_partitions_count
            return min_partitions_count
        
        return min_partitions(s, 0, {})