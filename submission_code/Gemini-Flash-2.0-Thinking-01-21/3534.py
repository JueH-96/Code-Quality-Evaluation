from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def is_almost_equal(x, y):
            if x == y:
                return True
            sx = str(x)
            sy = str(y)
            n = len(sx)
            m = len(sy)

            def check_swap(s1_str, s2_int):
                n1 = len(s1_str)
                for i in range(n1):
                    for j in range(n1):
                        s1_list = list(s1_str)
                        s1_list[i], s1_list[j] = s1_list[j], s1_list[i]
                        s1_swapped_str = "".join(s1_list)
                        try:
                            s1_swapped_int = int(s1_swapped_str)
                        except ValueError:
                            continue
                        if s1_swapped_int == s2_int:
                            return True
                return False

            if check_swap(sx, y):
                return True
            if check_swap(sy, x):
                return True
            return False

        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if is_almost_equal(nums[i], nums[j]):
                    count += 1
        return count