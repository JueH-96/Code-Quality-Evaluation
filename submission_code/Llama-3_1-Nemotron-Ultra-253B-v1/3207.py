class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        min_len = min(len(s1), len(s2), len(s3))
        min_ops = float('inf')
        
        for L in range(0, min_len + 1):
            if L == 0:
                if s1 == "" and s2 == "" and s3 == "":
                    sum_ops = 0
                    min_ops = min(min_ops, sum_ops)
            else:
                if len(s1) >= L and len(s2) >= L and len(s3) >= L:
                    prefix1 = s1[:L]
                    prefix2 = s2[:L]
                    prefix3 = s3[:L]
                    if prefix1 == prefix2 == prefix3:
                        sum_ops = (len(s1) - L) + (len(s2) - L) + (len(s3) - L)
                        min_ops = min(min_ops, sum_ops)
        
        return min_ops if min_ops != float('inf') else -1