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
                    # Check if x can be swapped to y
                    swapped_x = self.get_swapped_numbers(x)
                    if y in swapped_x:
                        count += 1
                    else:
                        # Check if y can be swapped to x
                        swapped_y = self.get_swapped_numbers(y)
                        if x in swapped_y:
                            count += 1
        return count
    
    def get_swapped_numbers(self, num: int) -> set:
        s = str(num)
        n = len(s)
        swapped = set()
        for i in range(n):
            for j in range(i + 1, n):
                chars = list(s)
                chars[i], chars[j] = chars[j], chars[i]
                swapped_num = int(''.join(chars))
                swapped.add(swapped_num)
        return swapped