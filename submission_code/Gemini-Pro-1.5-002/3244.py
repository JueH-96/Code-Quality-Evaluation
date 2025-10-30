class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 1
        
        while True:
            
            zeros = 0
            for num in nums:
                if num == 0:
                    zeros += 1
            
            if zeros >= n - 1:
                return n - zeros
            
            
            found = False
            for i in range(len(nums)):
                for j in range(len(nums)):
                    if i != j and nums[i] > 0 and nums[j] > 0:
                        
                        new_num = nums[i] % nums[j]
                        
                        new_nums = []
                        for k in range(len(nums)):
                            if k != i and k != j:
                                new_nums.append(nums[k])
                        new_nums.append(new_num)
                        
                        nums = new_nums
                        found = True
                        break
                if found:
                    break
            
            n = len(nums)