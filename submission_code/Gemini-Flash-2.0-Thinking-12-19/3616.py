class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        zero_indices = [i for i, num in enumerate(nums) if num == 0]
        count = 0

        for start_index in zero_indices:
            for direction in ['left', 'right']:
                current_nums = list(nums)
                curr = start_index
                current_direction = direction

                while 0 <= curr < n:
                    if current_nums[curr] == 0:
                        if current_direction == 'right':
                            curr += 1
                        else:
                            curr -= 1
                    elif current_nums[curr] > 0:
                        current_nums[curr] -= 1
                        current_direction = 'right' if current_direction == 'left' else 'left'
                        if current_direction == 'right':
                            curr += 1
                        else:
                            curr -= 1
                    else: # should not happen, but just in case
                        break
                
                all_zeros = True
                for num in current_nums:
                    if num != 0:
                        all_zeros = False
                        break
                if all_zeros:
                    count += 1
        return count