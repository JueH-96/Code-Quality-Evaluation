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
                # Check all possible swaps in x
                x_str = str(x)
                x_swapped = set()
                for a in range(len(x_str)):
                    for b in range(a + 1, len(x_str)):
                        lst = list(x_str)
                        lst[a], lst[b] = lst[b], lst[a]
                        new_num = int(''.join(lst))
                        x_swapped.add(new_num)
                if y in x_swapped:
                    count += 1
                    continue
                # Check all possible swaps in y
                y_str = str(y)
                y_swapped = set()
                for a in range(len(y_str)):
                    for b in range(a + 1, len(y_str)):
                        lst = list(y_str)
                        lst[a], lst[b] = lst[b], lst[a]
                        new_num = int(''.join(lst))
                        y_swapped.add(new_num)
                if x in y_swapped:
                    count += 1
        return count