class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        
        #Handle edge case where all numbers are the same except one
        if len(set(nums)) <=2:
            return nums[-1]

        potential_outliers = []
        for i in range(n):
            temp_nums = nums[:i] + nums[i+1:]
            
            #Check for duplicates to avoid invalid sums
            if len(set(temp_nums)) != len(temp_nums):
                continue

            s = sum(temp_nums)
            if s in nums and nums.index(s) != i:
                potential_outliers.append(nums[i])

        if not potential_outliers:
            #Handle edge case where no outlier is found using the above logic. This can happen if the outlier is the smallest number.
            for i in range(n):
                temp_nums = nums[:i] + nums[i+1:]
                s = sum(temp_nums)
                if s in nums and nums.index(s) != i:
                    potential_outliers.append(nums[i])
                    break

        return max(potential_outliers)