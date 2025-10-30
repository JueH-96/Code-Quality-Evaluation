class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for start_index in range(n):
            if nums[start_index] == 0:
                # Try moving left
                temp_nums_left = nums[:]
                curr_left = start_index
                direction_left = -1
                valid_left = True
                while True:
                    if curr_left < 0 or curr_left >= n:
                        break
                    if temp_nums_left[curr_left] == 0:
                        curr_left += direction_left
                    else:
                        temp_nums_left[curr_left] -= 1
                        direction_left *= -1
                        curr_left += direction_left
                if all(x == 0 for x in temp_nums_left):
                    count += 1

                # Try moving right
                temp_nums_right = nums[:]
                curr_right = start_index
                direction_right = 1
                valid_right = True
                while True:
                    if curr_right < 0 or curr_right >= n:
                        break
                    if temp_nums_right[curr_right] == 0:
                        curr_right += direction_right
                    else:
                        temp_nums_right[curr_right] -= 1
                        direction_right *= -1
                        curr_right += direction_right
                if all(x == 0 for x in temp_nums_right):
                    count += 1
        return count