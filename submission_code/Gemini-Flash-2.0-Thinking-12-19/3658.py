class Solution:
    def minDifference(self, nums: List[int]) -> int:
        def check(d):
            possible_x_range = [1, 10**9]
            possible_y_range = [1, 10**9]
            for i in range(1, len(nums)):
                u, v = nums[i-1], nums[i]
                if u != -1 and v != -1:
                    if abs(u - v) > d:
                        return False
                elif u == -1 and v != -1:
                    current_range = [max(1, v - d), min(10**9, v + d)]
                    possible_x_range[0] = max(possible_x_range[0], current_range[0])
                    possible_x_range[1] = min(possible_x_range[1], current_range[1])
                    possible_y_range[0] = max(possible_y_range[0], current_range[0])
                    possible_y_range[1] = min(possible_y_range[1], current_range[1])
                elif u != -1 and v == -1:
                    current_range = [max(1, u - d), min(10**9, u + d)]
                    possible_x_range[0] = max(possible_x_range[0], current_range[0])
                    possible_x_range[1] = min(possible_x_range[1], current_range[1])
                    possible_y_range[0] = max(possible_y_range[0], current_range[0])
                    possible_y_range[1] = min(possible_y_range[1], current_range[1])
            if possible_x_range[0] <= possible_x_range[1] or possible_y_range[0] <= possible_y_range[1]:
                return True
            return False

        low = 0
        high = 10**9
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans