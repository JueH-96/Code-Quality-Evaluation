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
                else:
                    if self.can_swap_to(x, y) or self.can_swap_to(y, x):
                        count += 1
        return count

    def can_swap_to(self, num: int, target: int) -> bool:
        s = str(num)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                lst = list(s)
                lst[i], lst[j] = lst[j], lst[i]
                new_num = int(''.join(lst))
                if new_num == target:
                    return True
        return False