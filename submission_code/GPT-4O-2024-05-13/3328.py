class Solution:
    def minOperations(self, k: int) -> int:
        nums = [1]
        current_sum = 1
        operations = 0
        
        while current_sum < k:
            max_num = max(nums)
            if current_sum + max_num <= k:
                nums.append(max_num)
                current_sum += max_num
                operations += 1
            else:
                nums[nums.index(max_num)] += 1
                current_sum += 1
                operations += 1
        
        return operations