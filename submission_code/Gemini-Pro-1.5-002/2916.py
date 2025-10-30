class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n == 1:
            return True

        for i in range(1, n):
            left_sum = sum(nums[:i])
            right_sum = sum(nums[i:])
            if (len(nums[:i]) == 1 or left_sum >= m) and (len(nums[i:]) == 1 or right_sum >= m):
                if self.check(nums[:i], m) and self.check(nums[i:], m):
                    return True
        return False

    def check(self, arr, m):
        n = len(arr)
        if n == 1:
            return True
        for i in range(1, n):
            left_sum = sum(arr[:i])
            right_sum = sum(arr[i:])

            if (len(arr[:i]) == 1 or left_sum >= m) and (len(arr[i:]) == 1 or right_sum >= m):
                if self.check(arr[:i], m) and self.check(arr[i:], m):
                    return True
        return False