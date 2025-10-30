from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        swap_sets = []
        for num in nums:
            s = str(num)
            n = len(s)
            current_set = set()
            for i in range(n):
                for j in range(i + 1, n):
                    lst = list(s)
                    lst[i], lst[j] = lst[j], lst[i]
                    new_num = int(''.join(lst))
                    current_set.add(new_num)
            swap_sets.append(current_set)
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] == nums[j]:
                    count += 1
                else:
                    if nums[j] in swap_sets[i] or nums[i] in swap_sets[j]:
                        count += 1
        return count