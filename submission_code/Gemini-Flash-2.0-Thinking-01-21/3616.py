class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        zero_indices = [i for i, num in enumerate(nums) if num == 0]

        for start_index in zero_indices:
            for direction in [-1, 1]: # -1 for left, 1 for right
                temp_nums = list(nums) # Create a copy
                curr = start_index
                current_direction = direction
                while 0 <= curr < n:
                    if temp_nums[curr] == 0:
                        curr += current_direction
                    else:
                        temp_nums[curr] -= 1
                        current_direction *= -1
                        curr += current_direction
                
                all_zeros = True
                for num in temp_nums:
                    if num != 0:
                        all_zeros = False
                        break
                if all_zeros:
                    count += 1
        return count