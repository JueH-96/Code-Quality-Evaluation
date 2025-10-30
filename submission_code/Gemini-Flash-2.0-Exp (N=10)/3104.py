class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0
        
        # Case 1: Select no students
        valid_no_selection = True
        for num in nums:
            if 0 >= num:
                valid_no_selection = False
                break
        if valid_no_selection:
            count += 1
        
        # Case 2: Select some students
        for i in range(1, 1 << n):
            selected_count = 0
            selected_indices = []
            for j in range(n):
                if (i >> j) & 1:
                    selected_count += 1
                    selected_indices.append(j)
            
            valid_selection = True
            for j in range(n):
                if (i >> j) & 1: # Student is selected
                    if selected_count <= nums[j]:
                        valid_selection = False
                        break
                else: # Student is not selected
                    if selected_count >= nums[j]:
                        valid_selection = False
                        break
            
            if valid_selection:
                count += 1
        
        return count