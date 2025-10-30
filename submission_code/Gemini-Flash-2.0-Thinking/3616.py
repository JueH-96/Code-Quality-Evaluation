class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        valid_selections_count = 0

        for start_index in range(n):
            if nums[start_index] == 0:
                # Try starting with left direction
                nums_copy_left = list(nums)
                curr_left = start_index
                direction_left = -1  # Left

                while 0 <= curr_left < n:
                    if nums_copy_left[curr_left] == 0:
                        curr_left += direction_left
                    else:
                        nums_copy_left[curr_left] -= 1
                        direction_left *= -1
                        curr_left += direction_left

                if all(x == 0 for x in nums_copy_left):
                    valid_selections_count += 1

                # Try starting with right direction
                nums_copy_right = list(nums)
                curr_right = start_index
                direction_right = 1  # Right

                while 0 <= curr_right < n:
                    if nums_copy_right[curr_right] == 0:
                        curr_right += direction_right
                    else:
                        nums_copy_right[curr_right] -= 1
                        direction_right *= -1
                        curr_right += direction_right

                if all(x == 0 for x in nums_copy_right):
                    valid_selections_count += 1

        return valid_selections_count