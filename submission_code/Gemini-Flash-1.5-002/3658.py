class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        missing = []
        for i in range(n):
            if nums[i] == -1:
                missing.append(i)

        if not missing:
            max_diff = 0
            for i in range(n - 1):
                max_diff = max(max_diff, abs(nums[i] - nums[i+1]))
            return max_diff

        if len(missing) == 0:
            max_diff = 0
            for i in range(n - 1):
                max_diff = max(max_diff, abs(nums[i] - nums[i+1]))
            return max_diff

        def calculate_diff(arr):
            max_diff = 0
            for i in range(len(arr) - 1):
                max_diff = max(max_diff, abs(arr[i] - arr[i+1]))
            return max_diff

        min_max_diff = float('inf')
        
        for x in range(1,101): # Iterate through possible values of x
            for y in range(x,101): # Iterate through possible values of y, ensuring y >=x
                temp_nums = nums[:]
                
                
                k=0
                for i in missing:
                    if k % 2 == 0:
                        temp_nums[i] = x
                    else:
                        temp_nums[i] = y
                    k+=1

                min_max_diff = min(min_max_diff, calculate_diff(temp_nums))

        return min_max_diff