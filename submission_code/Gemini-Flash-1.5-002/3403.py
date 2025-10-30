class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        count = {}
        for char in s:
            count[char] = count.get(char, 0) + 1
        
        def is_balanced(substring):
            substring_count = {}
            for char in substring:
                substring_count[char] = substring_count.get(char, 0) + 1
            
            first_count = None
            for char in substring_count:
                if first_count is None:
                    first_count = substring_count[char]
                elif substring_count[char] != first_count:
                    return False
            return True

        
        min_partitions = float('inf')
        
        def backtrack(index, current_partition):
            nonlocal min_partitions
            if index == len(s):
                min_partitions = min(min_partitions, len(current_partition))
                return

            for i in range(index + 1, len(s) + 1):
                substring = s[index:i]
                if is_balanced(substring):
                    backtrack(i, current_partition + [substring])

        backtrack(0, [])
        return min_partitions