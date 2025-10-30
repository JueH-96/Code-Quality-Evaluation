from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def can_almost_equal(a: int, b: int) -> bool:
            # If already equal, it's almost equal with zero swaps.
            if a == b:
                return True
            
            sa = list(str(a))
            sb = list(str(b))
            
            # Try swapping any two digits in a
            n = len(sa)
            for i in range(n):
                for j in range(i + 1, n):
                    sa[i], sa[j] = sa[j], sa[i]
                    if int("".join(sa)) == b:
                        return True
                    sa[i], sa[j] = sa[j], sa[i]
            
            # Try swapping any two digits in b
            m = len(sb)
            for i in range(m):
                for j in range(i + 1, m):
                    sb[i], sb[j] = sb[j], sb[i]
                    if int("".join(sb)) == a:
                        return True
                    sb[i], sb[j] = sb[j], sb[i]
            
            return False

        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if can_almost_equal(nums[i], nums[j]):
                    count += 1
        return count