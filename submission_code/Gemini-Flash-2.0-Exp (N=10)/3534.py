class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def is_almost_equal(x, y):
            x_str = str(x)
            y_str = str(y)
            if x_str == y_str:
                return True
            
            for i in range(len(x_str)):
                for j in range(i + 1, len(x_str)):
                    temp_x_list = list(x_str)
                    temp_x_list[i], temp_x_list[j] = temp_x_list[j], temp_x_list[i]
                    if "".join(temp_x_list) == y_str:
                        return True
            
            for i in range(len(y_str)):
                for j in range(i + 1, len(y_str)):
                    temp_y_list = list(y_str)
                    temp_y_list[i], temp_y_list[j] = temp_y_list[j], temp_y_list[i]
                    if "".join(temp_y_list) == x_str:
                        return True
            
            return False

        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if is_almost_equal(nums[i], nums[j]):
                    count += 1
        return count