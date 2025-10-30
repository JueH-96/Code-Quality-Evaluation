from itertools import permutations

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def is_almost_equal(x: int, y: int) -> bool:
            sx, sy = str(x), str(y)
            if sorted(sx) != sorted(sy):
                return False
            if len(sx) != len(sy):
                return True
            
            cnt = 0
            for a, b in zip(sx, sy):
                if a != b:
                    cnt += 1
                    if cnt > 2:
                        return False
            return True
        
        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if is_almost_equal(nums[i], nums[j]):
                    count += 1
        return count