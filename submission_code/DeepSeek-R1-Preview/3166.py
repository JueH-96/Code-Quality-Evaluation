from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: list[int]) -> int:
        cnt = list(Counter(nums).values())
        if not cnt:
            return 0
        
        m = min(cnt)
        min_total = float('inf')
        
        for s in range(m, 0, -1):
            total = 0
            valid = True
            for f in cnt:
                found = False
                b_max = f // (s + 1)
                for b in range(b_max, -1, -1):
                    rem = f - b * (s + 1)
                    if rem < 0:
                        continue
                    if rem % s == 0:
                        a = rem // s
                        total += a + b
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if valid:
                if total < min_total:
                    min_total = total
                # Early exit since we are trying s in descending order
                return min_total
        
        return min_total if min_total != float('inf') else 0