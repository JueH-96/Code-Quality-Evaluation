class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        valid = 0
        
        for start in range(n):
            if nums[start] != 0:
                continue
            for direction in [-1, 1]:
                curr = start
                dir_ = direction
                nums_copy = nums.copy()
                while True:
                    if curr < 0 or curr >= n:
                        break
                    if nums_copy[curr] == 0:
                        curr += dir_
                    else:
                        nums_copy[curr] -= 1
                        dir_ *= -1
                        curr += dir_
                if all(x == 0 for x in nums_copy):
                    valid += 1
        return valid