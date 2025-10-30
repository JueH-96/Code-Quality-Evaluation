class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            if nums[i] != 0:
                continue
            for direction in [1, -1]:
                current_nums = nums.copy()
                curr = i
                current_dir = direction
                sum_current = sum(current_nums)
                steps = 0
                valid = True
                while steps < sum_current:
                    if curr < 0 or curr >= n:
                        valid = False
                        break
                    if current_nums[curr] == 0:
                        curr += current_dir
                        steps += 1
                        if curr < 0 or curr >= n:
                            valid = False
                            break
                    else:
                        current_nums[curr] -= 1
                        sum_current -= 1
                        current_dir *= -1
                        curr += current_dir
                        steps += 1
                    if all(x == 0 for x in current_nums):
                        valid = True
                        break
                    if steps > 2 * sum_current:
                        valid = False
                        break
                if valid:
                    count += 1
        return count