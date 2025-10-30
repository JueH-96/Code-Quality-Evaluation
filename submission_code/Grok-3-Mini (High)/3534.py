from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        n = len(nums)
        # Precompute the set of values obtainable by swapping exactly two digits for each number
        swap_sets = []
        for num in nums:
            s = str(num)
            swap_set = set()
            len_s = len(s)
            if len_s >= 2:
                for i in range(len_s):
                    for j in range(i + 1, len_s):
                        # Create a copy of the string and swap the digits
                        s_list = list(s)
                        s_list[i], s_list[j] = s_list[j], s_list[i]
                        new_num = int(''.join(s_list))
                        swap_set.add(new_num)
            swap_sets.append(swap_set)
        
        # Count the number of pairs (i, j) with i < j that are almost equal
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                a = nums[i]
                b = nums[j]
                # They are almost equal if a == b, or b can be obtained by swapping digits in a,
                # or a can be obtained by swapping digits in b
                if a == b or b in swap_sets[i] or a in swap_sets[j]:
                    count += 1
        return count