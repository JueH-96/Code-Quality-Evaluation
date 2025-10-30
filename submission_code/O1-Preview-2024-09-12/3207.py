class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        min_operations = float('inf')
        pos = 0
        while True:
            # Check if all strings have length greater than pos
            if any(len(s_i) <= pos for s_i in [s1, s2, s3]):
                break
            c = s1[pos]
            if s2[pos] != c or s3[pos] != c:
                break
            len_W = pos + 1
            # Compute total deletions
            D_W = sum(len(s_i) - len_W for s_i in [s1, s2, s3])
            min_operations = min(min_operations, D_W)
            pos += 1
        if min_operations < float('inf'):
            return min_operations
        else:
            return -1