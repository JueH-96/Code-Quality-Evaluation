class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)

        def calculate_max_diff(arr):
            max_diff = 0
            for i in range(n - 1):
                max_diff = max(max_diff, abs(arr[i] - arr[i+1]))
            return max_diff

        def solve(x, y):
            temp_nums = []
            for num in nums:
                temp_nums.append(num)
            
            arr1 = []
            for num in temp_nums:
                arr1.append(num)

            for i in range(n):
                if arr1[i] == -1:
                    arr1[i] = x
            
            diff1 = calculate_max_diff(arr1)

            arr2 = []
            for num in temp_nums:
                arr2.append(num)

            for i in range(n):
                if arr2[i] == -1:
                    arr2[i] = y
            
            diff2 = calculate_max_diff(arr2)

            
            arr3 = []
            for num in temp_nums:
                arr3.append(num)
            
            def backtrack(index):
                if index == n:
                    return calculate_max_diff(arr3)
                
                if temp_nums[index] != -1:
                    return backtrack(index + 1)
                else:
                    arr3[index] = x
                    diff_x = backtrack(index + 1)
                    
                    arr3[index] = y
                    diff_y = backtrack(index + 1)
                    
                    arr3[index] = -1
                    return min(diff_x, diff_y)
            
            return backtrack(0)

        
        missing_indices = [i for i, num in enumerate(nums) if num == -1]
        if not missing_indices:
            return calculate_max_diff(nums)
        
        
        min_val = float('inf')
        max_val = float('-inf')
        
        for num in nums:
            if num != -1:
                min_val = min(min_val, num)
                max_val = max(max_val, num)
        
        if min_val == float('inf'):
            return 0
        
        
        x = (min_val + max_val) // 2
        y = x + 1
        
        return solve(x, y)