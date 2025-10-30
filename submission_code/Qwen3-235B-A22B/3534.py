from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                x = nums[i]
                y = nums[j]
                if x == y:
                    count += 1
                    continue
                # Check if swapping in x can get y, or swapping in y can get x
                if self.can_swap(x, y) or self.can_swap(y, x):
                    count += 1
        return count
    
    def can_swap(self, a: int, b: int) -> bool:
        s = list(str(a))
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                # Swap i and j
                new_s = s.copy()
                new_s[i], new_s[j] = new_s[j], new_s[i]
                swapped_num = int(''.join(new_s))
                if swapped_num == b:
                    return True
        return False