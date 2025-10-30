class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0

        def simulate(start, direction):
            arr = nums[:]
            curr = start
            dir = direction  # 1 for right, -1 for left

            while 0 <= curr < n:
                if arr[curr] == 0:
                    curr += dir
                else:
                    arr[curr] -= 1
                    dir *= -1
                    curr += dir

            for num in arr:
                if num != 0:
                    return False
            return True

        for i in range(n):
            if nums[i] == 0:
                if simulate(i, 1):
                    count += 1
                if simulate(i, -1):
                    count += 1

        return count