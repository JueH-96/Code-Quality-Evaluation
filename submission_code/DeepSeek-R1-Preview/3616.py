class Solution:
    def countValidSelections(self, nums):
        count = 0
        n = len(nums)
        for i in range(n):
            if nums[i] == 0:
                for direction in [-1, 1]:
                    current_nums = list(nums)
                    curr = i
                    dir_step = direction
                    while True:
                        if curr < 0 or curr >= n:
                            break
                        if current_nums[curr] == 0:
                            curr += dir_step
                        else:
                            current_nums[curr] -= 1
                            dir_step *= -1
                            curr += dir_step
                    if all(x == 0 for x in current_nums):
                        count += 1
        return count