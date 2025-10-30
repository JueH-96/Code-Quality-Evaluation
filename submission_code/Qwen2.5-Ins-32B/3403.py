class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        def is_balanced(sub):
            counts = {}
            for char in sub:
                counts[char] = counts.get(char, 0) + 1
            return len(set(counts.values())) == 1
        
        partitions = 1
        current_sub = ""
        
        for char in s:
            current_sub += char
            if not is_balanced(current_sub):
                partitions += 1
                current_sub = char
        
        return partitions