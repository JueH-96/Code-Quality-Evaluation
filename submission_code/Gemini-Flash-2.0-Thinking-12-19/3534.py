class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def are_almost_equal(x, y):
            if x == y:
                return True

            def get_swapped_numbers(n):
                s_n = str(n)
                swapped_nums = set()
                swapped_nums.add(n)
                n_len = len(s_n)
                for i in range(n_len):
                    for j in range(i + 1, n_len):
                        s_n_list = list(s_n)
                        s_n_list[i], s_n_list[j] = s_n_list[j], s_n_list[i]
                        swapped_num_str = "".join(s_n_list)
                        swapped_num = int(swapped_num_str)
                        swapped_nums.add(swapped_num)
                return swapped_nums

            swapped_x_nums = get_swapped_numbers(x)
            if y in swapped_x_nums:
                return True

            swapped_y_nums = get_swapped_numbers(y)
            if x in swapped_y_nums:
                return True
            
            return False

        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if are_almost_equal(nums[i], nums[j]):
                    count += 1
        return count