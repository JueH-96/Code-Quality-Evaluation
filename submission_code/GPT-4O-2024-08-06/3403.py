class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        from collections import Counter
        
        def is_balanced(counter):
            values = list(counter.values())
            return all(v == values[0] for v in values)
        
        n = len(s)
        min_partitions = 0
        i = 0
        
        while i < n:
            counter = Counter()
            for j in range(i, n):
                counter[s[j]] += 1
                if is_balanced(counter):
                    min_partitions += 1
                    i = j + 1
                    break
            else:
                # If we exit the loop without breaking, it means we reached the end
                # and the remaining part is balanced
                min_partitions += 1
                break
        
        return min_partitions