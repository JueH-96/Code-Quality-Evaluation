class Solution:
    def minDifference(self, nums: List[int]) -> int:
        def is_possible(max_diff):
            x_range = [1, 10**9 + 1]
            y_range = [1, 10**9 + 1]
            for i in range(len(nums) - 1):
                u, v = nums[i], nums[i+1]
                if u != -1 and v != -1:
                    if abs(u - v) > max_diff:
                        return False
                elif u == -1 and v != -1:
                    x_range[0] = max(x_range[0], v - max_diff)
                    x_range[1] = min(x_range[1], v + max_diff)
                    y_range[0] = max(y_range[0], v - max_diff)
                    y_range[1] = min(y_range[1], v + max_diff)
                elif u != -1 and v == -1:
                    x_range[0] = max(x_range[0], u - max_diff)
                    x_range[1] = min(x_range[1], u + max_diff)
                    y_range[0] = max(y_range[0], u - max_diff)
                    y_range[1] = min(y_range[1], u + max_diff)
            
            valid_x = x_range[0] <= x_range[1]
            valid_y = y_range[0] <= y_range[1]
            return valid_x or valid_y

        low = 0
        high = 10**9
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans