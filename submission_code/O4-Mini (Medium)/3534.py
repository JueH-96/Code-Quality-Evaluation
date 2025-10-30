from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def is_almost(x: int, y: int) -> bool:
            # If they're already equal, it's fine.
            if x == y:
                return True
            
            xs = list(str(x))
            ys = list(str(y))
            
            # Try swapping any two digits in x and see if it becomes y
            n = len(xs)
            for i in range(n):
                for j in range(i + 1, n):
                    xs[i], xs[j] = xs[j], xs[i]
                    if int("".join(xs)) == y:
                        return True
                    xs[i], xs[j] = xs[j], xs[i]  # swap back
            
            # Try swapping any two digits in y and see if it becomes x
            m = len(ys)
            for i in range(m):
                for j in range(i + 1, m):
                    ys[i], ys[j] = ys[j], ys[i]
                    if int("".join(ys)) == x:
                        return True
                    ys[i], ys[j] = ys[j], ys[i]  # swap back
            
            return False
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if is_almost(nums[i], nums[j]):
                    count += 1
        return count