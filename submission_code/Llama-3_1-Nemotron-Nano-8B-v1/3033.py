class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diffs = [i for i in range(n) if s1[i] != s2[i]]
        if len(diffs) % 2 != 0:
            return -1
        
        count = 0
        i = 0
        while i < len(diffs) - 1:
            if diffs[i + 1] == diffs[i] + 1:
                count += 1
                i += 2
            else:
                i += 1
        
        total_pairs = len(diffs) // 2
        return count * 1 + (total_pairs - count) * x