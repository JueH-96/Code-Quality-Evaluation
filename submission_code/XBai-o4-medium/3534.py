from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if self.check(nums[i], nums[j]):
                    count += 1
        return count
    
    def check(self, a: int, b: int) -> bool:
        sa = str(a)
        sb = str(b)
        max_len = max(len(sa), len(sb))
        a_padded = sa.zfill(max_len)
        b_padded = sb.zfill(max_len)
        
        if sorted(a_padded) != sorted(b_padded):
            return False
        
        if a_padded == b_padded:
            return True
        
        diffs = []
        for i in range(max_len):
            if a_padded[i] != b_padded[i]:
                diffs.append((a_padded[i], b_padded[i]))
        
        if len(diffs) == 2:
            return diffs[0][0] == diffs[1][1] and diffs[0][1] == diffs[1][0]
        return False