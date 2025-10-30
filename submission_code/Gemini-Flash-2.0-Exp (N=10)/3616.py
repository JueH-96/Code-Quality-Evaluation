class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0

        for start_index in range(n):
            if nums[start_index] != 0:
                continue
            
            for initial_direction in [-1, 1]: # -1 for left, 1 for right
                temp_nums = nums[:]
                curr = start_index
                direction = initial_direction
                
                valid = True
                
                while 0 <= curr < n:
                    if temp_nums[curr] == 0:
                        curr += direction
                    else:
                        temp_nums[curr] -= 1
                        direction *= -1
                        curr += direction
                
                if all(x == 0 for x in temp_nums):
                    count += 1
        
        return count